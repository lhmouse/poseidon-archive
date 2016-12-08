// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "url_param.hpp"

namespace Poseidon {

namespace Http {
	UrlParam::UrlParam(const OptionalMap &map_ref, const char *key)
		: m_valid(false), m_str()
	{
		const AUTO_REF(map, map_ref);
		const AUTO(it, map.find(SharedNts::view(key)));
		if(it != map.end()){
			m_valid = true;
			m_str = it->second;
		}
	}
	UrlParam::UrlParam(Move<OptionalMap> map_ref, const char *key)
		: m_valid(false), m_str()
	{
#ifdef POSEIDON_CXX11
		auto &map = map_ref;
#else
		OptionalMap map;
		map_ref.swap(map);
#endif
		const AUTO(it, map.find(SharedNts::view(key)));
		if(it != map.end()){
			m_valid = true;
			m_str.swap(it->second);
		}
#ifdef POSEIDON_CXX11
		// nothing
#else
		map_ref.swap(map);
#endif
	}
}

}
