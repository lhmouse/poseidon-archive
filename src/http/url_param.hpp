// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_URL_PARAM_HPP_
#define POSEIDON_HTTP_URL_PARAM_HPP_

#include <iosfwd>
#include <cstdlib>
#include "../optional_map.hpp"
#include "../uuid.hpp"

namespace Poseidon {

namespace Http {
	class UrlParam {
	public:
		bool m_valid;
		std::string m_str;

	public:
		UrlParam(Move<OptionalMap> map_ref, const char *key)
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

	public:
		bool valid() const NOEXCEPT {
			return m_valid;
		}
		const std::string &str() const NOEXCEPT {
			return m_str;
		}
		std::string &str() NOEXCEPT {
			return m_str;
		}

		const std::string &as_string() const NOEXCEPT {
			return m_str;
		}
		bool as_boolean() const NOEXCEPT {
			if(m_str.empty()){
				return false;
			}
			return m_str != "0";
		}
		long long as_signed() const NOEXCEPT {
			if(m_str.empty()){
				return 0;
			}
			return ::strtoll(m_str.c_str(), NULLPTR, 0);
		}
		unsigned long long as_unsigned() const NOEXCEPT {
			if(m_str.empty()){
				return 0;
			}
			return ::strtoull(m_str.c_str(), NULLPTR, 0);
		}
		double as_double() const NOEXCEPT {
			if(m_str.empty()){
				return 0;
			}
			return std::strtod(m_str.c_str(), NULLPTR);
		}
		Uuid as_uuid() const NOEXCEPT {
			Uuid uuid;
			if(!uuid.from_string(m_str)){
				return VAL_INIT;
			}
			return uuid;
		}

	public:
#ifdef POSEIDON_CXX11
		explicit operator bool() const noexcept {
			return valid();
		}
#else
		typedef bool (UrlParam::*DummyBool_)() const;
		operator DummyBool_() const NOEXCEPT {
			return valid() ? &UrlParam::valid : 0;
		}
#endif
		operator const std::string &() const NOEXCEPT {
			return str();
		}
		operator std::string &() NOEXCEPT {
			return str();
		}
	};

	inline std::ostream &operator<<(std::ostream &os, const UrlParam &rhs){
		return os << rhs.str();
	}
	inline std::istream &operator>>(std::istream &is, UrlParam &rhs){
		return is >> rhs.str();
	}
}

}

#endif
