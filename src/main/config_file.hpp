#ifndef POSEIDON_CONFIG_FILE_HPP_
#define POSEIDON_CONFIG_FILE_HPP_

#include "optional_map.hpp"

namespace Poseidon {

extern bool loadConfigFile(OptionalMap &ret, const char *path);
extern bool loadConfigFile(OptionalMap &ret, const std::string &path);

}

#endif
