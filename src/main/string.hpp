// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_STRING_HPP_
#define POSEIDON_STRING_HPP_

#include "cxx_ver.hpp"
#include "cxx_util.hpp"
#include <vector>
#include <string>
#include <cstddef>
#include <boost/lexical_cast.hpp>

namespace Poseidon {

// 参考 PHP 的 explode() 函数。
template<typename T>
inline std::vector<T> explode(char separator, const std::string &str, std::size_t limit = 0){
	std::vector<T> ret;
	if(!str.empty()){
		std::size_t begin = 0;
		std::string temp;
		for(;;){
			const std::size_t end = str.find(separator, begin);
			if(end == std::string::npos){
				temp.assign(str, begin, std::string::npos);
				ret.push_back(boost::lexical_cast<T>(temp));
				break;
			}
			if((limit != 0) && (ret.size() == limit - 1)){
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
		std::size_t begin = 0;
		std::string temp;
		for(;;){
			const std::size_t end = str.find(separator, begin);
			if(end == std::string::npos){
				temp.assign(str, begin, std::string::npos);
				ret.push_back(STD_MOVE(temp));
				break;
			}
			if((limit != 0) && (ret.size() == limit - 1)){
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
	std::ostringstream oss;
	for(AUTO(it, vec.begin()); it != vec.end(); ++it){
		oss <<*it;
		if(separator != 0){
			oss.put(separator);
		}
	}
	std::string ret = oss.str();
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
	return ret;
}

struct HexDumper {
	const void *const read;
	const std::size_t size;

	HexDumper(const void *read_, std::size_t size_)
		: read(read_), size(size_)
	{
	}
};

extern std::ostream &operator<<(std::ostream &os, const HexDumper &dumper);

inline std::string toUpperCase(std::string src){
	for(AUTO(it, src.begin()); it != src.end(); ++it){
		if(('a' <= *it) && (*it <= 'z')){
			*it -= 0x20;
		}
	}
	return STD_MOVE(src);
}
inline std::string toLowerCase(std::string src){
	for(AUTO(it, src.begin()); it != src.end(); ++it){
		if(('A' <= *it) && (*it <= 'Z')){
			*it += 0x20;
		}
	}
	return STD_MOVE(src);
}

}

#endif
