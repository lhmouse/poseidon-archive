// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_JSON_HPP_
#define POSEIDON_JSON_HPP_

#include "cxx_ver.hpp"
#include <string>
#include <iosfwd>
#include <cstddef>
#include <boost/container/flat_map.hpp>
#include <boost/container/vector.hpp>
#include <boost/variant.hpp>
#include <boost/type_traits/is_arithmetic.hpp>
#include <boost/type_traits/is_enum.hpp>
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

class JsonObject : public boost::container::flat_map<SharedNts, JsonElement> {
public:
	typedef boost::container::flat_map<SharedNts, JsonElement> BaseContainer;

public:
	JsonObject()
		: BaseContainer()
	{
	}
	JsonObject(const JsonObject &rhs)
		: BaseContainer(rhs)
	{
	}
	JsonObject &operator=(const JsonObject &rhs){
		JsonObject(rhs).swap(*this);
		return *this;
	}
#ifdef POSEIDON_CXX11
	JsonObject(JsonObject &&rhs) noexcept
		: BaseContainer(std::move(rhs))
	{
	}
	JsonObject &operator=(JsonObject &&rhs) noexcept {
		rhs.swap(*this);
		return *this;
	}
#endif

public:
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

class JsonArray : public boost::container::vector<JsonElement> {
public:
	typedef boost::container::vector<JsonElement> BaseContainer;

public:
	JsonArray()
		: BaseContainer()
	{
	}
	JsonArray(const JsonArray &rhs)
		: BaseContainer(rhs)
	{
	}
	JsonArray &operator=(const JsonArray &rhs){
		JsonArray(rhs).swap(*this);
		return *this;
	}
#ifdef POSEIDON_CXX11
	JsonArray(JsonArray &&rhs) noexcept
		: BaseContainer(std::move(rhs))
	{
	}
	JsonArray &operator=(JsonArray &&rhs) noexcept {
		rhs.swap(*this);
		return *this;
	}
#endif

public:
	void dump(std::string &str) const;
	std::string dump() const {
		std::string str;
		dump(str);
		return str;
	}
	void dump(std::ostream &os) const;
};

inline void swap(JsonArray &lhs, JsonArray &rhs) NOEXCEPT {
	lhs.swap(rhs);
}

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
	JsonElement(JsonNull = JsonNull())
		: m_data(JsonNull())
	{
	}
	JsonElement(bool rhs)
		: m_data(rhs)
	{
	}
	template<typename T>
	JsonElement(T rhs, typename boost::enable_if_c<
		boost::is_arithmetic<T>::value || boost::is_enum<T>::value,
		int>::type = 0)
		: m_data(static_cast<double>(rhs))
	{
	}
	JsonElement(const char *rhs)
		: m_data(std::string(rhs))
	{
	}
	JsonElement(std::string rhs)
		: m_data(STD_MOVE_IDN(rhs))
	{
	}
	JsonElement(JsonObject rhs)
		: m_data(STD_MOVE_IDN(rhs))
	{
	}
	JsonElement(JsonArray rhs)
		: m_data(STD_MOVE_IDN(rhs))
	{
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
	template<typename T>
	void set(T rhs){
		m_data(STD_MOVE_IDN(rhs)).swap(m_data);
	}

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
