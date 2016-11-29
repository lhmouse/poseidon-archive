// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "header_option.hpp"
#include "utilities.hpp"
#include "../string.hpp"

namespace Poseidon {

namespace Http {
	HeaderOption::HeaderOption(const std::string &str)
		: m_base(), m_options()
	{
		AUTO(pos, str.find(';'));
		if(pos == std::string::npos){
			m_base = trim(STD_MOVE(str));
		} else {
			AUTO(seg, str.substr(0, pos));
			++pos;
			m_base = trim(STD_MOVE(seg));
			for(;;){
				const AUTO(end, str.find(';', pos));
				if(end == std::string::npos){
					seg = str.substr(pos);
				} else {
					seg = str.substr(pos, end - pos);
				}
				std::string key;
				const AUTO(equ, seg.find('='));
				if(equ == std::string::npos){
					key = STD_MOVE(seg);
					seg.clear();
				} else {
					key = seg.substr(0, equ);
					seg.erase(0, equ + 1);
				}
				key = trim(STD_MOVE(key));
				if(!key.empty()){
					seg = trim(STD_MOVE(seg));
					if(!seg.empty() && (*seg.begin() == '\"') && (*seg.rbegin() == '\"')){
						seg.erase(seg.end() - 1);
						seg.erase(seg.begin());
						seg = trim(STD_MOVE(seg));
					}
					m_options.set(SharedNts(key), STD_MOVE(seg));
				}
				if(end == std::string::npos){
					break;
				}
				pos = end + 1;
			}
		}
	}
	HeaderOption::HeaderOption(std::string base, OptionalMap options)
		: m_base(STD_MOVE(base)), m_options(STD_MOVE(options))
	{
	}

	std::string HeaderOption::to_string() const {
		std::string ret;
		ret = m_base;
		for(AUTO(it, m_options.begin()); it != m_options.end(); ++it){
			ret += ';';
			ret += it->first.get();
			if(!it->second.empty()){
				ret += '=';
				ret += url_encode(it->second);
			}
		}
		return ret;
	}
}

}
