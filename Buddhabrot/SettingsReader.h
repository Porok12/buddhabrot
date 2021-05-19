#include <fstream>
#include <iostream>
#include "json.hpp"

using json = nlohmann::json;

namespace ns {
	struct settings {

	};
}

json load_config(std::string file = "config.json");
