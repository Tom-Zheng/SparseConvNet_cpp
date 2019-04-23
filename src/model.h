#ifndef MODEL_H_
#define MODEL_H_

#include <torch/torch.h>
#include "sparseconvnet/SCN/sparseconvnet.h"
#include <string>
#include <vector>
#include <map>

class Sequential : public torch::nn::Sequential
{
public:
    Sequential();
    template<typename ModuleType>
    Sequential& add(ModuleType module);
};

struct UNet : torch::nn::Module {
public:
    UNet();
    torch::Tensor forward(torch::Tensor coords, torch::Tensor features);

private:
	torch::Device *_device;
	Sequential seq;
	// torch::nn::Sequential seq;
};

#endif
