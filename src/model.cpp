#include "model.h"

UNet:UNet(const char *conf_file, torch::Device *device) {
    load_cfg(cfg_file);
	_device = device;
	create_modules();
}

void UNet:load_cfg(const char *cfg_file)  {
    ifstream fs(cfg_file);
	string line;
	if(!fs) 
	{
		std::cout << "Fail to load cfg file:" << cfg_file << endl;
		return;
	}
	while (getline (fs, line))
	{ 
		trim(line);
		if (line.empty())
		{
			continue;
		}		

		if ( line.substr (0,1)  == "[")
		{
			map<string, string> block;			

			string key = line.substr(1, line.length() -2);
			block["type"] = key;  

			blocks.push_back(block);
		}
		else
		{
			map<string, string> *block = &blocks[blocks.size() -1];
			vector<string> op_info;

			split(line, op_info, "=");

			if (op_info.size() == 2)
			{
				string p_key = op_info[0];
				string p_value = op_info[1];
				block->operator[](p_key) = p_value;
			}			
		}				
	}
	fs.close();
}