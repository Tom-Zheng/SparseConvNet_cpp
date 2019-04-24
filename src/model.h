#ifndef MODEL_H_
#define MODEL_H_

#include <torch/torch.h>
#include "sparseconvnet/SCN/sparseconvnet.h"
#include <string>
#include <vector>
#include <map>

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


class Sequential : public torch::nn::Module
{
public:
    Sequential();
    template<typename ModuleType>
    Sequential& add(ModuleType module);

    Sequential& add();

    SparseConvNetTensor forward(SparseConvNetTensor input);

    torch::nn::Sequential _modules;
};

struct UNet : torch::nn::Module {
public:
    UNet();

    torch::Tensor forward(torch::Tensor coords, torch::Tensor features);
    
    int load();

private:
	torch::Device *_device;
	// Sequential seq;
	torch::nn::Sequential seq;
};

#endif
