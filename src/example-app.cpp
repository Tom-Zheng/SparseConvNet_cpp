#include <torch/torch.h>
#include "model.h"

#include <iostream>
#include <memory>

#include "utils.h"

#include <cnpy.h>

int main() {
  long test_coord_[] = {
    15, 7, 0, 0,
    15, 8, 0, 0,
    3, 11, 0, 0,
    4, 12, 0, 0,
    7, 8, 0, 0,
    0, 15, 0, 0
  };
  torch::Tensor test_coord = torch::from_blob(test_coord_, {6,4}, torch::dtype(torch::kInt64));
  
  torch::Tensor test_features = torch::randn({6,3}, torch::dtype(torch::kFloat32));

  float test_weights[] = {
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
  };
  
  auto model = UNet();
  auto output = model.forward(test_coord, test_features);
  
  std::cout << output << std::endl;
  
  // model.load();

#if 0
  cnpy::NpyArray arr_ = cnpy::npy_load("/home/zheng/Desktop/cpp/scn_cpp/temp/arr.npy");
  torch::Tensor arr = torch::from_blob(arr_.data<float>(), {3,3,3}, torch::dtype(torch::kFloat32));
  
  std::cout << arr << std::endl;
#endif

  return 0;
}