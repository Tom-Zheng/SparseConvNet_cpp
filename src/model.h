#include <torch/torch.h>
#include <string>
#include <vector>
#include <map>

struct UNet : torch::nn::Module {
public:
    UNet(const char *conf_file, torch::Device *device);
    torch::Tensor forward(torch::Tensor x);

private:
	torch::Device *_device;
	vector<map<string, string>> blocks;
	torch::nn::Sequential features;
	vector<torch::nn::Sequential> module_list;

    void load_cfg(const char *cfg_file);
    void create_modules();
    int get_int_from_cfg(map<string, string> block, string key, int default_value);
    string get_string_from_cfg(map<string, string> block, string key, string default_value);
};
