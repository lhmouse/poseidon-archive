#ifndef POSEIDON_CONFIG_FILE_HPP_
#define POSEIDON_CONFIG_FILE_HPP_

#include <string>

namespace Poseidon {

struct ConfigFile {
	static void reload(const char *path);

	static const std::string &get(const char *key);
	static const std::string &get(const std::string &key);

private:
	ConfigFile();
};

}

#endif
