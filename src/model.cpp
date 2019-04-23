#include "model.h"
#include "utils.h"
#include <stdio.h>
#include <iostream>
#include <typeinfo>
#include <cmath>
#include <memory>
#include <tuple>
#include <string>
#include <vector>

/**
 * SparseConvNetTensor
*/
class SparseConvNetTensor{
public:
    torch::Tensor features;     /*FloatTensor*/
    std::shared_ptr<Metadata<3>> metadata;
    torch::Tensor spatial_size; /*LongTensor*/

    SparseConvNetTensor()   {
    }

    SparseConvNetTensor(torch::Tensor features_, 
                        std::shared_ptr<Metadata<3>> metadata_, 
                        torch::Tensor spatial_size_)   {
        features = features_;
        metadata = metadata_;
        spatial_size = spatial_size_;
    }

    torch::Tensor cuda()  {
        return features.cuda();
    }

    torch::Tensor cpu()  {
        return features.cpu();
    }
};

/**
 * Submanifold Convolution Layers
*/
class Identity : public torch::nn::Module
{
public:
    Identity(){
    }

    SparseConvNetTensor forward(SparseConvNetTensor input) {
        return input; 
    }
};


class SubmanifoldConvolution : public torch::nn::Module
{
public:
    int dimension;
    int nIn;
    int nOut;
    torch::Tensor filter_size;
    long filter_volume;
    torch::Tensor weight;

    SubmanifoldConvolution(int dimension_, 
                           int nIn_, 
                           int nOut_, 
                           int filter_size_, 
                           bool bias_) {
        dimension = dimension_;
        nIn = nIn_;
        nOut = nOut_;
        filter_size = toLongTensor(dimension, filter_size_);
        filter_volume = filter_size.prod().item().toLong();
        double std = sqrt((2.0 / nIn / filter_volume));
        weight = this->register_parameter("weight", torch::empty({filter_volume, nIn, nOut}).normal_(0, std).cuda());
        if(bias_)  {
            // Not implemented yet
        }
    }

    SparseConvNetTensor forward(SparseConvNetTensor input) {
        assert(input.features.size(0) == 0 || input.features.size(1) == nIn);
        SparseConvNetTensor output;
        output.metadata = input.metadata;
        output.spatial_size = input.spatial_size;
        
        auto options = input.features.options();
        
        output.features = torch::empty(0, options);

        SubmanifoldConvolution_updateOutput<3>(
            input.spatial_size,
            filter_size,
            *(input.metadata),
            input.features,
            output.features,
            weight,
            torch::empty(0,options));

        return output;
    }
};

class InputLayer : public torch::nn::Module   {
public:
    int dimension;
    torch::Tensor spatial_size;
    long mode;
    torch::Device device;

    InputLayer(int dimension_, 
                torch::Tensor spatial_size_,
                int mode_ = 3) : device("cuda"){
        dimension = dimension_;
        spatial_size = spatial_size_;
        mode = (long)mode_;
    }

    SparseConvNetTensor forward(std::tuple<torch::Tensor, torch::Tensor> input)   {
        
        // torch::Tensor coords, torch::Tensor features

        // auto options = torch::TensorOptions()
        //                 .device(torch::kCUDA, 1)
        //                 .dtype(torch::kFloat32);
        torch::Tensor coords = std::get<0>(input);
        torch::Tensor features = std::get<1>(input);

        auto options = features.options();

        SparseConvNetTensor output(
            torch::empty(0, options).to(device),
            std::make_shared<Metadata<3>>(),
            spatial_size
        );

        // std::cout << spatial_size << std::endl;
        // std::cout << coords.cpu().toType(torch::kInt64) << std::endl;
        // std::cout << features.to(device) << std::endl;

        InputLayer_updateOutput<3>(
            *(output.metadata),
            spatial_size,
            coords.cpu().toType(torch::kInt64),
            features.to(device),
            output.features,
            0,
            mode
        );
        return output;
    }
};

class OutputLayer : public torch::nn::Module  {
public:
    int dimension;

    OutputLayer(int dimension_) {
        dimension = dimension_;
    }

    torch::Tensor forward(SparseConvNetTensor input)   {
        auto options = input.features.options();
        auto output = torch::empty(0, options);
        OutputLayer_updateOutput<3>(
            *(input.metadata),
            input.features.contiguous(),
            output
        );
        return output;
    }
};

class BatchNormLeakyReLU : public torch::nn::Module
{
public:
    int nPlanes;
    double eps;
    double momentum;
    double leakiness;

    torch::Tensor running_mean;
    torch::Tensor running_var;
    torch::Tensor weight;
    torch::Tensor bias;

    BatchNormLeakyReLU(int nPlanes_, 
                       double eps_ = 1e-4, 
                       double momentum_=0.9,
                       double leakiness_=0.333)
    {
        nPlanes = nPlanes_;
        eps = eps_;
        momentum = momentum_;
        leakiness = leakiness_;

        running_mean = register_buffer("running_mean", torch::empty(nPlanes).fill_(0));
        running_var = register_buffer("running_var", torch::empty(nPlanes).fill_(1));

        weight = register_parameter("weight", torch::empty(nPlanes).fill_(1).cuda());
        bias = register_parameter("bias", torch::empty(nPlanes).fill_(0).cuda());
    }

    SparseConvNetTensor forward(SparseConvNetTensor input) {
        assert(input.features.size(0) == 0 || input.features.size(1) == nPlanes);

        SparseConvNetTensor output;
        output.metadata = input.metadata;
        output.spatial_size = input.spatial_size;
        
        auto options = input.features.options();
        output.features = torch::empty(0, options);

        torch::Tensor unused_saveMean = torch::empty(nPlanes, options);
        torch::Tensor unused_saveInvStd = torch::empty(nPlanes, options);

        BatchNormalization_updateOutput(
            input.features,
            output.features,
            unused_saveMean,
            unused_saveInvStd,
            running_mean,
            running_var,
            weight,
            bias,
            eps,
            momentum,
            false,
            leakiness);
        
        return output;
    }
};


class ConcatTable : public torch::nn::Sequential
{
public:
    ConcatTable() {}

    template<typename ModuleType>
    ConcatTable& add(ModuleType module) {
        this->get()->push_back(module);
        return *this;
    }
    
    template<typename ReturnType = SparseConvNetTensor>
    std::vector<ReturnType> forward(SparseConvNetTensor input)  
    {
        std::vector<ReturnType> output;
        auto iterator = this->get()->begin();
        for (++iterator; iterator != this->get()->end(); ++iterator) {
            output.push_back( iterator->any_forward(std::move(input)));
        }
    }
};


class JoinTable : public torch::nn::Sequential
{
public:
    JoinTable() {}

    template<typename ModuleType>
    JoinTable& add(ModuleType module) {
        this->get()->push_back(module);
        return *this;
    }

    template<typename ReturnType = SparseConvNetTensor, 
            typename InputType = std::vector<SparseConvNetTensor>>
    ReturnType forward(InputType input)  
    {
        SparseConvNetTensor output;
        output.metadata = input[0].metadata;
        output.spatial_size = input[0].spatial_size;
        
        auto options = input.features.options();

        output.features = torch::empty(0, options);

        // output.features = torch.cat([i.features for i in input], 1) if input[0].features.numel() else input[0].features
        
        for(auto iter : input)  {
            output.features = torch::cat(output.features, *iter);
        }

        return output;
    }
};

Sequential::Sequential() {}

template<typename ModuleType>
Sequential& Sequential::add(ModuleType module) {
    this->get()->push_back(module);
    return *this;
}

UNet::UNet() {
    long spatial_size[] = {20, 20, 20};
    torch::Tensor spatial_size_tensor = torch::from_blob(spatial_size, {3}, torch::dtype(torch::kInt64));

    int dimension = 3;

    seq = Sequential().add(
        InputLayer(dimension, spatial_size_tensor, 4)
    ).add(
        ConcatTable()
        .add(
            Identity()
        )
        .add(
            Identity()
        )
    ).add(
        JoinTable()
    ).add(
        OutputLayer(dimension)
    );
    
    std::string name = "sparsemodel";
    register_module(name, seq);
}

torch::Tensor UNet::forward(torch::Tensor coords, torch::Tensor features){
    auto input = std::make_tuple(coords, features);
    return seq->forward(input);
}

void load() {
    // TODO
}
