// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_JSON_WRITER_HPP_
#define POSEIDON_JSON_WRITER_HPP_

#include "cxx_ver.hpp"
#include <deque>
#include <string>
#include <map>
#include <iosfwd>
#include <cstddef>
#include <boost/variant.hpp>
#include <boost/type_traits/is_arithmetic.hpp>
#include <boost/utility/enable_if.hpp>
#include "shared_nts.hpp"

namespace Poseidon {

class JsonElement;

class JsonObject : public std::map<SharedNts, JsonElement> {
public:
	void dump(std::ostream &os) const;
	std::string dump() const;
};

class JsonArray : public std::deque<JsonElement> {
public:
	void dump(std::ostream &os) const;
	std::string dump() const;
};

class JsonElement {
private:
	boost::variant<
		bool,			// 0
		long double,	// 1
		std::string,	// 2
		JsonObject,		// 3
		JsonArray		// 4
		> m_data;

public:
	JsonElement()
		: m_data(false)
	{
	}
	template<typename T>
	JsonElement(T t){
		set(STD_MOVE(t));
	}

public:
	void set(bool rhs){
		m_data = rhs;
	}
	template<typename T>
	void set(T rhs, typename boost::enable_if_c<boost::is_arithmetic<T>::value>::type * = NULLPTR){
		m_data = static_cast<long double>(rhs);
	}
	void set(const char *rhs){
		m_data = std::string(rhs);
	}
	void set(std::string rhs){
		m_data = std::string(STD_MOVE(rhs));
	}
	void set(JsonObject rhs){
		m_data = JsonObject(STD_MOVE(rhs));
	}
	void set(JsonArray rhs){
		m_data = JsonArray(STD_MOVE(rhs));
	}
	void set(JsonElement rhs){
		m_data.swap(rhs.m_data);
	}
#ifndef POSEIDON_CXX11
	template<typename T>
	void set(Move<T> rhs){
		set(T(rhs));
	}
#endif

	void swap(JsonElement &rhs) NOEXCEPT {
		m_data.swap(rhs.m_data);
	}

	void dump(std::ostream &os) const;
	std::string dump() const;
};

inline void swap(JsonObject &lhs, JsonObject &rhs) NOEXCEPT {
	lhs.swap(rhs);
}
inline std::ostream &operator<<(std::ostream &os, const JsonObject &rhs){
	rhs.dump(os);
	return os;
}

inline void swap(JsonArray &lhs, JsonArray &rhs) NOEXCEPT {
	lhs.swap(rhs);
}
inline std::ostream &operator<<(std::ostream &os, const JsonArray &rhs){
	rhs.dump(os);
	return os;
}

inline void swap(JsonElement &lhs, JsonElement &rhs) NOEXCEPT {
	lhs.swap(rhs);
}
inline std::ostream &operator<<(std::ostream &os, const JsonElement &rhs){
	rhs.dump(os);
	return os;
}

}

#endif
