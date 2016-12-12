// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_STRING_HPP_
#define POSEIDON_STRING_HPP_

#include "cxx_ver.hpp"
#include "cxx_util.hpp"
#include "buffer_streams.hpp"
#include <vector>
#include <string>
#include <cstddef>
#include <boost/lexical_cast.hpp>

namespace Poseidon {

extern const std::string &empty_string() NOEXCEPT;

// 参考 PHP 的 explode() 函数。
template<typename T>
inline std::vector<T> explode(char separator, const std::string &str, std::size_t limit = 0){
	std::vector<T> ret;
	if(!str.empty()){
		std::size_t begin = 0, end;
		std::string temp;
		for(;;){
			if((limit != 0) && (ret.size() == limit - 1)){
				end = std::string::npos;
			} else {
				end = str.find(separator, begin);
			}
			if(end == std::string::npos){
				temp.assign(str, begin, std::string::npos);
				ret.push_back(boost::lexical_cast<T>(temp));
				break;
			}
			temp.assign(str, begin, end - begin);
			ret.push_back(boost::lexical_cast<T>(temp));
			begin = end + 1;
		}
	}
	return ret;
}
template<>
inline std::vector<std::string> explode(char separator, const std::string &str, std::size_t limit){
	std::vector<std::string> ret;
	if(!str.empty()){
		std::size_t begin = 0, end;
		std::string temp;
		for(;;){
			if((limit != 0) && (ret.size() == limit - 1)){
				end = std::string::npos;
			} else {
				end = str.find(separator, begin);
			}
			if(end == std::string::npos){
				temp.assign(str, begin, std::string::npos);
				ret.push_back(STD_MOVE(temp));
				break;
			}
			temp.assign(str, begin, end - begin);
			ret.push_back(STD_MOVE(temp));
			begin = end + 1;
		}
	}
	return ret;
}

template<typename T>
inline std::string implode(char separator, const std::vector<T> &vec){
	Buffer_ostream os;
	for(AUTO(it, vec.begin()); it != vec.end(); ++it){
		os <<*it;
		if(separator != 0){
			os <<separator;
		}
	}
	std::string ret = os.get_buffer().dump_string();
	if(!ret.empty()){
		ret.erase(ret.end() - 1);
	}
	return ret;
}
template<>
inline std::string implode(char separator, const std::vector<std::string> &vec){
	std::string ret;
	for(AUTO(it, vec.begin()); it != vec.end(); ++it){
		ret.append(*it);
		if(separator != 0){
			ret.push_back(separator);
		}
	}
	if(!ret.empty()){
		ret.erase(ret.end() - 1);
	}
	return ret;
}

inline std::string to_upper_case(std::string src){
	for(AUTO(it, src.begin()); it != src.end(); ++it){
		if(('a' <= *it) && (*it <= 'z')){
			*it -= 0x20;
		}
	}
	return STD_MOVE(src);
}
inline std::string to_lower_case(std::string src){
	for(AUTO(it, src.begin()); it != src.end(); ++it){
		if(('A' <= *it) && (*it <= 'Z')){
			*it += 0x20;
		}
	}
	return STD_MOVE(src);
}

inline std::string ltrim(std::string src){
	const AUTO(pos, src.find_first_not_of(" \t"));
	src.erase(0, pos);
	return STD_MOVE(src);
}
inline std::string rtrim(std::string src){
	const AUTO(pos, src.find_last_not_of(" \t"));
	src.erase(pos + 1);
	return STD_MOVE(src);
}
inline std::string trim(std::string src){
	return ltrim(rtrim(STD_MOVE(src)));
}

extern bool is_valid_utf8_string(const std::string &str);

}

#endif
