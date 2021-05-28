#include <fstream>
#include <iostream>
#include "json.hpp"

using json = nlohmann::json;

/**
 * @brief wczytanie pliku konfiguracyjnego
 * @param file œcie¿ka do pliku
 * @return obiekt json zawieraj¹cy dane z pliku lub pusty w przypadku braku pliku lub z³ego formatu
*/
json load_config(std::string file = "config.json");
