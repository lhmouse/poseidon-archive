// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "config_file.hpp"
#include "file.hpp"
#include "string.hpp"
#include "log.hpp"
#include "system_exception.hpp"

namespace Poseidon {

ConfigFile::ConfigFile(){
}
ConfigFile::ConfigFile(const char *path){
	load(path);
}

void ConfigFile::load(const char *path){
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Loading config file: ", path);

	StreamBuffer buffer;
	fileGetContents(buffer, path);

	OptionalMap contents;
	std::string line;
	std::size_t count = 0;
	while(getLine(buffer, line)){
		++count;
		std::size_t pos = line.find('#');
		if(pos != std::string::npos){
			line.erase(pos);
		}
		pos = line.find_first_not_of(" \t");
		if(pos == std::string::npos){
			continue;
		}
		std::size_t equ = line.find('=', pos);
		if(equ == std::string::npos){
			LOG_POSEIDON_ERROR("Error in config file on line ", count, ": '=' expected.");
			DEBUG_THROW(Exception, sslit("Bad config file"));
		}

		std::size_t keyEnd = line.find_last_not_of(" \t", equ - 1);
		if((keyEnd == std::string::npos) || (pos >= keyEnd)){
			LOG_POSEIDON_ERROR("Error in config file on line ", count, ": Name expected.");
			DEBUG_THROW(Exception, sslit("Bad config file"));
		}
		SharedNts key(line.data() + pos, static_cast<std::size_t>(keyEnd + 1 - pos));

		std::string val;
		pos = line.find_first_not_of(" \t", equ + 1);
		if(pos != std::string::npos){
			val.assign(line, pos, std::string::npos);
			pos = val.find_last_not_of(" \t");
			val.erase(pos + 1);
		}

		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG, "Config: ", key, " = ", val);
		contents.append(STD_MOVE(key), STD_MOVE(val));
	}
	m_contents.swap(contents);
}

int ConfigFile::loadNoThrow(const char *path){
	try {
		load(path);
		return 0;
	} catch(SystemException &e){
		return e.code();
	}
}

}
