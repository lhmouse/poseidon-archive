// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "config_file.hpp"
#include "profiler.hpp"
#include "log.hpp"
#include "string.hpp"
#include "buffer_streams.hpp"
#include "singletons/filesystem_daemon.hpp"
#include "system_exception.hpp"

namespace Poseidon {

namespace {
	void escape(std::ostream &os, const char *data, std::size_t size){
		PROFILE_ME;

		for(std::size_t i = 0; i < size; ++i){
			const unsigned ch = (unsigned char)data[i];
			switch(ch){
			case '\b':
				os <<'\\' <<'b';
				break;
			case '\f':
				os <<'\\' <<'f';
				break;
			case '\n':
				os <<'\\' <<'n';
				break;
			case '\r':
				os <<'\\' <<'r';
				break;
			case '\t':
				os <<'\\' <<'t';
				break;
			case '\\':
			case ' ':
			case '=':
			case '#':
				os <<'\\' <<(char)ch;
				break;
			default:
				os <<(char)ch;
				break;
			}
		}
	}
	char unescape(std::string &seg, std::istream &is, const char *terminators){
		PROFILE_ME;

		char term;
		seg.clear();
		bool escaped = false;
		for(;;){
			char ch;
			if(!is.get(ch)){
				term = 0;
				break;
			}
			if(escaped){
				switch(ch){
				case '\b':
					seg += '\b';
					break;
				case '\f':
					seg += '\f';
					break;
				case '\n':
					seg += '\n';
					break;
				case '\r':
					seg += '\r';
					break;
				case '\t':
					seg += '\t';
					break;
				default:
					seg += ch;
					break;
				}
				escaped = false;
			} else if(ch == '\\'){
				escaped = true;
			} else {
				const char *const pos = std::strchr(terminators, ch);
				if(pos){
					term = *pos;
					break;
				}
				seg += ch;
			}
		}
		return term;
	}
}

void ConfigFile::load(const std::string &path){
	PROFILE_ME;
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Loading config file: ", path);

	AUTO(block, FileSystemDaemon::load(path));
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Read ", block.size_total, " byte(s) from ", path);

	VALUE_TYPE(m_contents) contents;

	std::size_t line = 0;
	for(;;){
		StreamBuffer buf;
		for(;;){
			const int ch = block.data.get();
			if(ch < 0){
				break;
			}
			if(ch == '\n'){
				break;
			}
			buf.put((unsigned char)ch);
		}
		if(buf.back() == '\r'){
			buf.unput();
		}
		if(buf.empty() && block.data.empty()){
			break;
		}
		++line;

		Buffer_istream is(STD_MOVE(buf));
		std::string key, val;
		const char key_term = unescape(key, is, "=#");
		key = trim(STD_MOVE(key));
		if(key.empty()){
			continue;
		}
		if(key_term == '='){
			unescape(val, is, "#");
			val = trim(STD_MOVE(val));
		}
		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG, "Config: #", std::setw(3), line, " | ", key, " = ", val);
		contents.append(SharedNts(key), STD_MOVE(val));
	}

	m_contents.swap(contents);
}
int ConfigFile::load_nothrow(const std::string &path)
try {
	PROFILE_ME;

	load(path);
	return 0;
} catch(SystemException &e){
	LOG_POSEIDON_ERROR("SystemException thrown while loading config file: path = ", path, ", code = ", e.get_code(), ", what = ", e.what());
	return e.get_code();
} catch(std::exception &e){
	LOG_POSEIDON_ERROR("std::exception thrown while loading config file: path = ", path, ", what = ", e.what());
	return EINVAL;
} catch(...){
	LOG_POSEIDON_ERROR("Unknown exception thrown while loading config file: path = ", path);
	return EINVAL;
}

void ConfigFile::save(const std::string &path){
	PROFILE_ME;
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Saving config file: ", path);

	Buffer_ostream os;
	for(AUTO(it, m_contents.begin()); it != m_contents.end(); ++it){
		escape(os, it->first.get(), std::strlen(it->first.get()));
		os <<" = ";
		escape(os, it->second.data(), it->second.size());
		os <<std::endl;
	}
	FileSystemDaemon::save(path, STD_MOVE(os.get_buffer()));
}

}
