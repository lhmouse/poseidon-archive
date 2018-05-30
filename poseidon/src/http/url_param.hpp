// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_URL_PARAM_HPP_
#define POSEIDON_HTTP_URL_PARAM_HPP_

#include <iosfwd>
#include <cstdlib>
#include "../option_map.hpp"
#include "../uuid.hpp"

namespace Poseidon {
namespace Http {

class Url_param {
public:
	bool m_valid;
	std::string m_str;

public:
	Url_param(const Option_map &map_ref, const char *key);
	Url_param(Move<Option_map> map_ref, const char *key);

public:
	bool valid() const NOEXCEPT {
		return m_valid;
	}
	const std::string & str() const NOEXCEPT {
		return m_str;
	}
	std::string & str() NOEXCEPT {
		return m_str;
	}

	const std::string & as_string() const NOEXCEPT {
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
	typedef bool (Url_param::*Dummy_bool_)() const;
	operator Dummy_bool_() const NOEXCEPT {
		return valid() ? &Url_param::valid : 0;
	}
#endif

	operator const std::string &() const NOEXCEPT {
		return str();
	}
	operator std::string &() NOEXCEPT {
		return str();
	}
};

inline std::ostream & operator<<(std::ostream &os, const Url_param &rhs){
	return os << rhs.str();
}

}
}

#endif
