// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_JSON_HPP_
#define POSEIDON_JSON_HPP_

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

#ifdef POSEIDON_CXX11
using JsonNull = std::nullptr_t;
#else
struct JsonNull {
	void *unused;
};
#endif

class JsonElement;

class JsonObject : public std::map<SharedNts, JsonElement> {
public:
	void dump(std::string &str) const;
	std::string dump() const {
		std::string str;
		dump(str);
		return str;
	}
	void dump(std::ostream &os) const;
};

class JsonArray : public std::deque<JsonElement> {
public:
	void dump(std::string &str) const;
	std::string dump() const {
		std::string str;
		dump(str);
		return str;
	}
	void dump(std::ostream &os) const;
};

class JsonElement {
public:
	enum Type {
		T_BOOL, T_NUMBER, T_STRING, T_OBJECT, T_ARRAY, T_NULL
	};

private:
	boost::variant<
		bool, double, std::string, JsonObject, JsonArray, JsonNull
		> m_data;

public:
	JsonElement()
		: m_data(JsonNull())
	{
	}
	template<typename T>
	JsonElement(T t){
		set(STD_MOVE_IDN(t));
	}

public:
	Type type() const {
		return static_cast<Type>(m_data.which());
	}

	template<typename T>
	const T &get() const {
		return boost::get<const T &>(m_data);
	}
	template<typename T>
	T &get(){
		return boost::get<T &>(m_data);
	}

	void set(JsonNull){
		m_data = JsonNull();
	}
	void set(bool rhs){
		m_data = rhs;
	}
	template<typename T>
	void set(T rhs, typename boost::enable_if_c<boost::is_arithmetic<T>::value>::type * = NULLPTR){
		m_data = static_cast<double>(rhs);
	}
	void set(const char *rhs){
		m_data = std::string(rhs);
	}
	void set(std::string rhs){
		m_data = STD_MOVE_IDN(rhs);
	}
	void set(JsonObject rhs){
		m_data = STD_MOVE_IDN(rhs);
	}
	void set(JsonArray rhs){
		m_data = STD_MOVE_IDN(rhs);
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

	void dump(std::string &str) const;
	std::string dump() const {
		std::string str;
		dump(str);
		return str;
	}
	void dump(std::ostream &os) const;
};

inline void swap(JsonObject &lhs, JsonObject &rhs) NOEXCEPT {
	lhs.swap(rhs);
}
inline void swap(JsonArray &lhs, JsonArray &rhs) NOEXCEPT {
	lhs.swap(rhs);
}
inline void swap(JsonElement &lhs, JsonElement &rhs) NOEXCEPT {
	lhs.swap(rhs);
}

inline std::ostream &operator<<(std::ostream &os, const JsonObject &rhs){
	rhs.dump(os);
	return os;
}
inline std::ostream &operator<<(std::ostream &os, const JsonArray &rhs){
	rhs.dump(os);
	return os;
}
inline std::ostream &operator<<(std::ostream &os, const JsonElement &rhs){
	rhs.dump(os);
	return os;
}

class JsonParser {
private:
	static std::string accept_string(std::istream &is);
	static double accept_number(std::istream &is);
	static JsonObject accept_object(std::istream &is);
	static JsonArray accept_array(std::istream &is);
	static bool accept_boolean(std::istream &is);
	static JsonNull accept_null(std::istream &is);

public:
	static JsonElement parse_element(std::istream &is);
	static JsonObject parse_object(std::istream &is);
	static JsonArray parse_array(std::istream &is);

private:
	JsonParser();
};

inline std::istream &operator>>(std::istream &is, JsonElement &rhs){
	JsonParser::parse_element(is).swap(rhs);
	return is;
}
inline std::istream &operator>>(std::istream &is, JsonArray &rhs){
	JsonParser::parse_array(is).swap(rhs);
	return is;
}
inline std::istream &operator>>(std::istream &is, JsonObject &rhs){
	JsonParser::parse_object(is).swap(rhs);
	return is;
}

}

#endif
