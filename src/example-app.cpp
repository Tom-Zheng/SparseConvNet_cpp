#include <torch/torch.h>
#include "model.h"
#include <iostream>
#include <memory>

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
  
  float test_features_[] = {
        -1.3784e+00, -5.8401e-01,  9.7296e-02,
         4.1258e-02,  1.1464e+00, -2.5578e-01,
        -1.9137e-01, -1.7091e-03, -1.7312e+00,
         1.2702e+00, -1.7412e+00, -7.0317e-01,
        -9.0240e-01, -1.0003e+00,  8.7477e-01,
         1.7333e-01, -1.9368e-02, -7.6304e-01
  };
  
  torch::Tensor test_features = torch::from_blob(test_features_, {6,3}, torch::dtype(torch::kFloat32));

  // std::string dir = "";
  std::string dir = "/home/zheng/Desktop/SparseConvNet_training/examples/ScanNet/weight";

  auto model = UNet(dir);
  auto output = model.forward(test_coord, test_features);
  
  std::cout << output << std::endl;

  return 0;
}
