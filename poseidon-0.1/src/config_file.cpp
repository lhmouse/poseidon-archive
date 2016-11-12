// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

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
	file_get_contents(buffer, path);

	OptionalMap contents;
	std::string raw_line, line;
	std::size_t count = 0;
	while(get_line(buffer, raw_line)){
		++count;

		line.clear();
		line.reserve(raw_line.size());
		bool escaped = false;
		for(AUTO(it, raw_line.begin()); it != raw_line.end(); ++it){
			char ch = *it;
			if(escaped){
				escaped = false;
				if(ch == 'b'){
					ch = '\b';
				} else if(ch == 'f'){
					ch = '\f';
				} else if(ch == 'n'){
					ch = '\n';
				} else if(ch == 'r'){
					ch = '\r';
				} else if(ch == 't'){
					ch = '\t';
				}
				line.push_back(ch);
			} else if(ch == '\\'){
				escaped = true;
			} else {
				// escaped = false;
				if(ch == '#'){
					break;
				}
				line.push_back(ch);
			}
		}

		std::size_t pos = line.find_first_not_of(" \t");
		if(pos == std::string::npos){
			continue;
		}
		std::size_t equ = line.find('=', pos);
		if(equ == std::string::npos){
			LOG_POSEIDON_ERROR("Error in config file on line ", count, ": '=' expected.");
			DEBUG_THROW(Exception, sslit("Bad config file"));
		}

		std::size_t key_end = line.find_last_not_of(" \t", equ - 1);
		if((key_end == std::string::npos) || (pos >= key_end)){
			LOG_POSEIDON_ERROR("Error in config file on line ", count, ": Name expected.");
			DEBUG_THROW(Exception, sslit("Bad config file"));
		}
		SharedNts key(line.data() + pos, static_cast<std::size_t>(key_end + 1 - pos));

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

int ConfigFile::load_nothrow(const char *path){
	try {
		load(path);
		return 0;
	} catch(SystemException &e){
		return e.get_code();
	}
}

}
