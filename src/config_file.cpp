// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "config_file.hpp"
#include "string.hpp"
#include "profiler.hpp"
#include "log.hpp"
#include "buffer_streams.hpp"
#include "singletons/filesystem_daemon.hpp"
#include "system_exception.hpp"

namespace Poseidon {

namespace {
	std::string unescape_line(const char *data, std::size_t size){
		PROFILE_ME;

		std::string line;
		line.reserve(size);
		for(std::size_t i = 0; i < size; ++i){
			const char ch = data[i];
			switch(ch){
			case '\b':
				line += '\\';
				line += '\b';
				break;
			case '\f':
				line += '\\';
				line += '\f';
				break;
			case '\n':
				line += '\\';
				line += '\n';
				break;
			case '\r':
				line += '\\';
				line += '\r';
				break;
			case '\t':
				line += '\\';
				line += '\t';
				break;
			case '\\':
				line += '\\';
				line += '\\';
				break;
			case '#':
				line += '\\';
				line += '#';
				break;
			default:
				line += ch;
				break;
			}
		}
		return line;
	}
	std::string escape_line(const char *data, std::size_t size){
		PROFILE_ME;

		std::string line;
		line.reserve(size);
		bool escaped = false;
		for(std::size_t i = 0; i < size; ++i){
			const char ch = data[i];
			if(escaped){
				switch(ch){
				case 'b':
					line += '\b';
					break;
				case 'f':
					line += '\f';
					break;
				case 'n':
					line += '\n';
					break;
				case 'r':
					line += '\r';
					break;
				case 't':
					line += '\t';
					break;
				case '\\':
					line += '\\';
					break;
				default:
					line += ch;
					break;
				}
				escaped = false;
			} else if(ch == '\\'){
				escaped = true;
			} else if(ch == '#'){
				break;
			} else {
				line += ch;
				// escaped = false;
			}
		}
		return line;
	}
}

void ConfigFile::load(const std::string &path){
	PROFILE_ME;
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Loading config file: ", path);

	AUTO(block, FileSystemDaemon::load(path));
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Read ", block.size_total, " byte(s) from ", path);

	OptionalMap contents;

	Buffer_istream bis(STD_MOVE(block.data));
	std::string line;
	std::size_t count = 0;
	while(std::getline(bis, line)){
		if(!line.empty() && (*line.rbegin() == '\r')){
			line.erase(line.end() - 1);
		}
		line = escape_line(line.data(), line.size());
		++count;

		std::size_t key_begin = line.find_first_not_of(" \t");
		if(key_begin == std::string::npos){
			continue;
		}
		std::size_t equ = line.find('=', key_begin);
		if(equ == std::string::npos){
			LOG_POSEIDON_ERROR("Error in config file on line ", count, ": '=' expected.");
			DEBUG_THROW(Exception, sslit("Bad config file"));
		}
		std::size_t key_end = line.find_last_not_of(" \t", equ - 1);
		if((key_end == std::string::npos) || (key_begin > key_end)){
			LOG_POSEIDON_ERROR("Error in config file on line ", count, ": Name expected.");
			DEBUG_THROW(Exception, sslit("Bad config file"));
		}
		++key_end;
		SharedNts key(line.data() + key_begin, static_cast<std::size_t>(key_end - key_begin));
		line.erase(0, equ + 1);
		std::string value(trim(STD_MOVE(line)));
		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG, "Config: ", key, " = ", value);
		contents.append(STD_MOVE(key), STD_MOVE(value));
	}

	m_contents.swap(contents);
}
int ConfigFile::load_nothrow(const std::string &path){
	PROFILE_ME;

	try {
		load(path);
		return 0;
	} catch(SystemException &e){
		return e.get_code();
	}
}
void ConfigFile::save(const std::string &path){
	PROFILE_ME;
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Saving config file: ", path);

	Buffer_ostream os;
	for(AUTO(it, m_contents.begin()); it != m_contents.end(); ++it){
		os <<unescape_line(it->first.get(), std::strlen(it->first.get()))
		   <<" = "
		   <<unescape_line(it->second.data(), it->second.size())
		   <<std::endl;
	}
	FileSystemDaemon::save(path, STD_MOVE(os.get_buffer()));
}

}
