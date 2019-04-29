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
#include <fstream>
#include <sstream>
#include <cnpy.h>

/*
* Load PATH
*/

std::string path;
int weight_count = 0;

torch::Tensor LoadNpy(std::string layer_name, torch::Tensor weight) {
    string s = path + std::string("/") + layer_name + std::string(".npy");
    std::cout << std::string("Loading") + s << std::endl;

    cnpy::NpyArray arr_ = cnpy::npy_load(s);
    c10::IntList shape((long*)arr_.shape.data(), arr_.shape.size());

    if(shape != weight.sizes()) {
        std::cout << "Loaded shape:" << std::endl;
        std::cout << arr_.shape << std::endl;
        std::cout << "Expected shape:" << std::endl;
        std::cout << weight.sizes() << std::endl;
        abort();
    }

    return torch::from_blob(arr_.data<float>(), shape, torch::dtype(torch::kFloat32)).cuda();
}


/**
 * Submanifold Convolution Layers
*/
class Identity : public torch::nn::Module
{
public:
    Identity(){

    }

    SparseConvNetTensor forward(SparseConvNetTensor input) {
        SparseConvNetTensor output;
        output.metadata = input.metadata;
        output.spatial_size = input.spatial_size;
        output.features = input.features;

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

    std::string layer_name;

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
        
        std::cout << "SubmanifoldConvolution " << nIn << " -> " << nOut << std::endl;

        layer_name = std::string("unet_") + std::to_string(weight_count++);

        if(path.length())   {
            weight = LoadNpy(layer_name, weight);
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


class Convolution : public torch::nn::Module
{
public:
    int dimension;
    int nIn;
    int nOut;
    torch::Tensor filter_size;
    long filter_volume;
    torch::Tensor filter_stride;
    
    torch::Tensor weight;

    std::string layer_name;

    Convolution(int dimension_, 
                int nIn_, 
                int nOut_, 
                int filter_size_, 
                int filter_stride_, 
                bool bias_) {
        dimension = dimension_;
        nIn = nIn_;
        nOut = nOut_;
        filter_size = toLongTensor(dimension, filter_size_);
        filter_volume = filter_size.prod().item().toLong();
        filter_stride = toLongTensor(dimension, filter_stride_);
        double std = sqrt((2.0 / nIn / filter_volume));
        weight = this->register_parameter("weight", torch::empty({filter_volume, nIn, nOut}).normal_(0, std).cuda());
        if(bias_)  {
            // Not implemented yet
        }

        std::cout << "Convolution " << nIn << " -> " << nOut << std::endl;

        layer_name = std::string("unet_") + std::to_string(weight_count++);

        if(path.length())   {
            weight = LoadNpy(layer_name, weight);
        }
    }

    SparseConvNetTensor forward(SparseConvNetTensor input) {
        assert(input.features.size(0) == 0 || input.features.size(1) == nIn);
        SparseConvNetTensor output;
        output.metadata = input.metadata;
        output.spatial_size = (input.spatial_size - filter_size) / filter_stride + 1;
        
        assert((((output.spatial_size - 1) * filter_stride + filter_size) == input.spatial_size).all().item().toInt());

        // if((((output.spatial_size - 1) * filter_stride + filter_size) == input.spatial_size).all().item().toInt() == 0) {
        //     std::cout << "input.spatial_size" << std::endl;
        //     std::cout << input.spatial_size << std::endl;
        //     std::cout << "output.spatial_size" << std::endl;
        //     std::cout << output.spatial_size << std::endl;
        //     abort();
        // }
        
        auto options = input.features.options();
        output.features = torch::empty(0, options);

        Convolution_updateOutput<3>(
            input.spatial_size,
            output.spatial_size,
            filter_size,
            filter_stride,
            *(input.metadata),
            input.features,
            output.features,
            weight,
            torch::empty(0,options));

        return output;
    }
};

class Deconvolution : public torch::nn::Module
{
public:
    int dimension;
    int nIn;
    int nOut;
    torch::Tensor filter_size;
    long filter_volume;
    torch::Tensor filter_stride;
    
    torch::Tensor weight;

    std::string layer_name;

    Deconvolution(int dimension_, 
                int nIn_, 
                int nOut_, 
                int filter_size_, 
                int filter_stride_, 
                bool bias_) {
        dimension = dimension_;
        nIn = nIn_;
        nOut = nOut_;
        filter_size = toLongTensor(dimension, filter_size_);
        filter_volume = filter_size.prod().item().toLong();
        filter_stride = toLongTensor(dimension, filter_stride_);
        double std = sqrt((2.0 / nIn / filter_volume));
        weight = this->register_parameter("weight", torch::empty({filter_volume, nIn, nOut}).normal_(0, std).cuda());
        if(bias_)  {
            // Not implemented yet
        }

        std::cout << "Deconvolution " << nIn << " -> " << nOut << std::endl;

        layer_name = std::string("unet_") + std::to_string(weight_count++);

        if(path.length())   {
            weight = LoadNpy(layer_name, weight);
        }
    }

    SparseConvNetTensor forward(SparseConvNetTensor input) {
        assert(input.features.size(0) == 0 || input.features.size(1) == nIn);
        SparseConvNetTensor output;
        output.metadata = input.metadata;
        output.spatial_size = (input.spatial_size - 1) * filter_stride + filter_size;
        
        auto options = input.features.options();
        output.features = torch::empty(0, options);

        Deconvolution_updateOutput<3>(
            input.spatial_size,
            output.spatial_size,
            filter_size,
            filter_stride,
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
        
        std::cout << "InputLayer" << std::endl;
        std::cout << output.spatial_size << std::endl;

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

    std::string weight_name;
    std::string bias_name;
    std::string rm_name;
    std::string rv_name;

    BatchNormLeakyReLU(int nPlanes_, 
                       double eps_ = 1e-4, 
                       double momentum_=0.9,
                       double leakiness_=0)
    {
        nPlanes = nPlanes_;
        eps = eps_;
        momentum = momentum_;
        leakiness = leakiness_;

        running_mean = register_buffer("running_mean", torch::empty(nPlanes).fill_(0).cuda());
        running_var = register_buffer("running_var", torch::empty(nPlanes).fill_(1).cuda());

        weight = register_parameter("weight", torch::empty(nPlanes).fill_(1).cuda());
        bias = register_parameter("bias", torch::empty(nPlanes).fill_(0).cuda());
    
        weight_name = std::string("unet_") + std::to_string(weight_count++);
        bias_name = std::string("unet_") + std::to_string(weight_count++);
        rm_name = std::string("unet_") + std::to_string(weight_count++);
        rv_name = std::string("unet_") + std::to_string(weight_count++);

        std::cout << "BatchNormLeakyReLU " << nPlanes << std::endl;

        if(path.length())   {
            weight = LoadNpy(weight_name, weight);
            bias = LoadNpy(bias_name, bias);
            running_mean = LoadNpy(rm_name, running_mean);
            running_var = LoadNpy(rv_name, running_var);
        }
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
            true,
            leakiness);
        
        return output;
    }
};

class ConcatTable : public torch::nn::Module
{
public:
    ConcatTable() {}

    template<typename ModuleType>
    ConcatTable& add(ModuleType module)
    {
        _modules->push_back(module);
        return *this;
    }

    SparseConvNetTensor forward(SparseConvNetTensor input)  
    {
        // Forward for each submodule
        std::vector<SparseConvNetTensor> values;
        auto iterator = _modules->begin();
        for (; iterator != _modules->end(); ++iterator) {
            values.push_back( iterator->forward<SparseConvNetTensor>(input));
        }
        
        // Join
        SparseConvNetTensor output;
        output.metadata = values[0].metadata;
        output.spatial_size = values[0].spatial_size;

        // output.features = torch.cat([i.features for i in input], 1) if input[0].features.numel() else input[0].features
        auto iterator_vector = values.begin();
        output.features = iterator_vector->features;
        for (++iterator_vector; iterator_vector != values.end(); ++iterator_vector) {

            output.features = torch::cat({output.features, iterator_vector->features}, 1);
        }
        
        return output;
    }

    torch::nn::Sequential _modules;
};

class AddTable : public torch::nn::Module
{
public:
    AddTable() {}

    template<typename ModuleType>
    AddTable& add(ModuleType module)
    {
        _modules->push_back(module);
        return *this;
    }

    SparseConvNetTensor forward(SparseConvNetTensor input)  
    {
        // Forward for each submodule
        std::vector<SparseConvNetTensor> values;
        auto iterator = _modules->begin();
        for (; iterator != _modules->end(); ++iterator) {
            values.push_back( iterator->forward<SparseConvNetTensor>(input));
        }
        
        // AddTable
        SparseConvNetTensor output;
        output.metadata = values[0].metadata;
        output.spatial_size = values[0].spatial_size;

        auto iterator_vector = values.begin();
        output.features = iterator_vector->features;
        for (++iterator_vector; iterator_vector != values.end(); ++iterator_vector) {

            output.features += iterator_vector->features;
        }
        
        return output;
    }

    torch::nn::Sequential _modules;
};

class NetworkInNetwork : public torch::nn::Module
{
public:
    int nIn;
    int nOut;
    torch::Tensor weight;
    std::string weight_name;

    NetworkInNetwork(int nIn_, 
                    int nOut_,
                    bool bias_)
    {
        nIn = nIn_;
        nOut = nOut_;

        double std = sqrt((2.0 / nIn));
        weight = this->register_parameter("weight", torch::empty({nIn, nOut}).normal_(0, std).cuda());

        weight_name = std::string("unet_") + std::to_string(weight_count++);

        if(bias_)  {
            // Not implemented yet
        }

        std::cout << "NetworkInNetwork " << nIn << " -> " << nOut << std::endl;

        if(path.length())   {
            weight = LoadNpy(weight_name, weight);
        }
    }

    SparseConvNetTensor forward(SparseConvNetTensor input) {
        assert(input.features.size(0) == 0 || input.features.size(1) == nIn);
        SparseConvNetTensor output;
        output.metadata = input.metadata;
        output.spatial_size = input.spatial_size;
        
        auto options = input.features.options();
        output.features = torch::empty(0, options);

        NetworkInNetwork_updateOutput(
            input.features,
            output.features,
            weight,
            torch::empty(0,options));
        
        return output;
    }
};



// The original torch::nn::Sequential does not support nesting due to templatized return type,
// here is a little hack to support nesting by specifying return type.

Sequential::Sequential() {}

Sequential& Sequential::add() {return *this;}

template<typename ModuleType>
Sequential& Sequential::add(ModuleType module) {
    _modules->push_back(module);
    return *this;
}

SparseConvNetTensor Sequential::forward(SparseConvNetTensor input)  {
    return _modules->forward<SparseConvNetTensor>(input);
}


/*
* Recursive function
*/
const int m = 16;
const int unet_depth = 7;

Sequential unet_block(int a, int b, bool residual_block) {
    if(residual_block)  {
        AddTable block;
        if(a == b)
            block.add(Identity());
        else
            block.add(NetworkInNetwork(a,b,false));
        Sequential foo;
        foo.add(BatchNormLeakyReLU(a));
        foo.add(SubmanifoldConvolution(3, a, b, 3, false));
        foo.add(BatchNormLeakyReLU(b));
        foo.add(SubmanifoldConvolution(3, b, b, 3, false));

        block.add(foo);
        return Sequential().add(block);
    } else  {
        Sequential block;
        block.add(BatchNormLeakyReLU(a));
        block.add(SubmanifoldConvolution(3, a, b, 3, false));
        return block;
    }
}

Sequential unet_build(int depth, bool residual_block)  {
    Sequential seq;
    seq.add(unet_block(m*depth, m*depth, residual_block));
    if(depth < unet_depth) {
        Sequential s;
        s.add(BatchNormLeakyReLU(m*depth));
        s.add(Convolution(3, m*depth, m*(depth+1), 2, 2, false));
        s.add(unet_build(depth+1, residual_block));
        s.add(BatchNormLeakyReLU(m*(depth+1)));
        s.add(Deconvolution(3, m*(depth+1), m*depth, 2, 2, false));
        
        ConcatTable c;
        c.add(Identity());
        c.add(s);

        seq.add(c);
                
        seq.add(unet_block(2*m*depth, m*depth, residual_block));
    }
    return seq;
}

UNet::UNet(std::string weights_path) 
{
    long spatial_size[] = {4096, 4096, 4096};
    torch::Tensor spatial_size_tensor = torch::from_blob(spatial_size, {3}, torch::dtype(torch::kInt64));
    
    int dimension = 3;
    
    path = weights_path;
    // Recursively build unet
#if 0

    auto linear = torch::nn::Linear(torch::nn::LinearOptions(m, 20).with_bias(true));
    linear->to(torch::kCUDA);

    seq = torch::nn::Sequential(
        InputLayer(dimension, spatial_size_tensor, 4),
        Sequential().add(
            SubmanifoldConvolution(dimension, 1, 1, 3, false)
        ),
        OutputLayer(dimension)
        // linear
    );
#endif

#if 1
    auto linear = torch::nn::Linear(torch::nn::LinearOptions(m, 20).with_bias(true));
    linear->to(torch::kCUDA);
    
    Sequential body;
    body.add(SubmanifoldConvolution(dimension, 3, m, 3, false));
    body.add(unet_build(1, true));
    body.add(BatchNormLeakyReLU(m));

    seq = torch::nn::Sequential(
        InputLayer(dimension, spatial_size_tensor.clone(), 4),
        body,
        OutputLayer(dimension),
        linear
    );

    // Load linear layer
    if(path.length() > 0)   {
        linear->weight = LoadNpy(std::string("unet_") + std::to_string(weight_count++), linear->weight);
        linear->bias = LoadNpy(std::string("unet_") + std::to_string(weight_count++), linear->bias);
    }

#endif

    std::string name = "sparsemodel";
    register_module(name, seq);
}

torch::Tensor UNet::forward(torch::Tensor coords, torch::Tensor features){
    auto input = std::make_tuple(coords, features);
    return seq->forward(input);
}
