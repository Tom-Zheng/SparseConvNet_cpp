#include <torch/torch.h>
#include "sparseconvnet/SCN/sparseconvnet.h"

#include <iostream>
#include <memory>

int main() {
  // Torch API
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;
  // SparseConvNet
  auto metadata = std::make_shared<Metadata<3>>();
  metadata->clear();
  return 0;
}