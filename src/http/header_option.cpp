// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "header_option.hpp"
#include "urlencoded.hpp"
#include "../string.hpp"
#include "../profiler.hpp"
#include "../buffer_streams.hpp"

namespace Poseidon {

namespace Http {
	std::string HeaderOption::dump() const {
		PROFILE_ME;

		Buffer_ostream os;
		dump(os);
		return os.get_buffer().dump_string();
	}
	void HeaderOption::dump(std::ostream &os) const {
		PROFILE_ME;

		os <<m_base;
		for(AUTO(it, m_options.begin()); it != m_options.end(); ++it){
			os <<';';
			os <<' ';
			os <<it->first.get();
			if(!it->second.empty()){
				os <<'=';
				url_encode(os, it->second);
			}
		}
	}
	void HeaderOption::parse(std::istream &is){
		PROFILE_ME;

		VALUE_TYPE(m_base) base;
		VALUE_TYPE(m_options) options;

		std::string seg;
		std::size_t count = 0;
		typedef std::istream::traits_type traits;
		traits::int_type next = is.peek();
		for(;;){
			seg.clear();
			bool quoted = false;
			for(; !traits::eq_int_type(next, traits::eof()); next = is.peek()){
				const char ch = is.get();
				if(quoted){
					if(ch == '\"'){
						quoted = false;
						continue;
					}
					seg += ch;
					continue;
				}
				if(ch == '\"'){
					quoted = true;
					continue;
				}
				if(ch == ';'){
					break;
				}
				seg += ch;
			}
			if(seg.empty()){
				if(traits::eq_int_type(next, traits::eof())){
					break;
				}
				continue;
			}
			++count;

			if(count == 1){
				base = trim(STD_MOVE(seg));
			} else {
				std::size_t key_begin = seg.find_first_not_of(" \t");
				if(key_begin == std::string::npos){
					continue;
				}
				std::size_t equ = seg.find('=', key_begin);
				std::size_t key_end;
				if(equ == std::string::npos){
					key_end = seg.find_last_not_of(" \t");
				} else {
					key_end = seg.find_last_not_of(" \t", equ - 1);
				}
				if((key_end == std::string::npos) || (key_begin > key_end)){
					continue;
				}
				++key_end;
				SharedNts key(seg.data() + key_begin, static_cast<std::size_t>(key_end - key_begin));
				if(equ == std::string::npos){
					seg.clear();
				} else {
					seg.erase(0, equ + 1);
				}
				std::string value(trim(STD_MOVE(seg)));
				seg.clear();
				options.append(STD_MOVE(key), STD_MOVE(value));
			}
		}

		m_base.swap(base);
		m_options.swap(options);
	}
}

}
