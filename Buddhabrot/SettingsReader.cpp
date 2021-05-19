#include "SettingsReader.h"

json load_config(std::string file) {
	std::ifstream config_file(file);
	if (config_file) {
		json j;
		try {
			config_file >> j;
			return j;
		}
		catch (json::parse_error e) {
			std::cout << "Json corrupted!" << std::endl;
		}
	}
	else {
		std::cout << "File not found!" << std::endl;
	}

	return json();
}

/*json j = load_config("config.json");
if (j.contains("iterations")) {

	std::cout << j["iterations"].value("red", 10) << std::endl;
	std::cout << j["iterations"].value("green", 10) << std::endl;
	std::cout << j["iterations"].value("blue", 10) << std::endl;
}
//std::cout << j.value("layer-color", 10) << std::endl;
//std::cout << j.value("layer-color", 10) << std::endl;
//std::cout << j.value("layer-color", 10) << std::endl;
if (j.contains("gamma")) {
	std::cout << j.value("red", 10) << std::endl;
	std::cout << j.value("green", 10) << std::endl;
	std::cout << j.value("blue", 10) << std::endl;
}
std::cout << j.value("c.re", 0) << std::endl;
std::cout << j.value("c.im", 0) << std::endl;
std::cout << j.value("scale", 0) << std::endl;
std::cout << j.value("rotation", 0) << std::endl;
std::cout << j.value("z.re", 0) << std::endl;
std::cout << j.value("z.im", 0) << std::endl;
std::cout << j.value("samples", 1000000) << std::endl;
std::cout << j.value("width", 1280) << std::endl;
std::cout << j.value("height", 720) << std::endl;
if (j.contains("background")) {
	std::cout << j.value("red", 10) << std::endl;
	std::cout << j.value("green", 10) << std::endl;
	std::cout << j.value("blue", 10) << std::endl;
}
std::cout << j.value("blending", false) << std::endl;
std::cout << j.value("output-file", "buddhabrot") << std::endl;*/