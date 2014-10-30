#ifndef POSEIDON_SINGLETONS_MAIN_CONFIG_HPP_
#define POSEIDON_SINGLETONS_MAIN_CONFIG_HPP_

#include "../config_file.hpp"

namespace Poseidon {

struct MainConfig {
	static void setRunPath(const SharedNtmbs &path);

	static void reload();
	static const ConfigFile &getConfigFile();

private:
	MainConfig();
};

}

#endif
