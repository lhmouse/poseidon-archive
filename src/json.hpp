// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_JSON_HPP_
#define POSEIDON_JSON_HPP_

#include "cxx_ver.hpp"
#include <string>
#include <iosfwd>
#include <stdexcept>
#include <cstddef>
#include <boost/container/map.hpp>
#include <boost/container/deque.hpp>
#include <boost/variant.hpp>
#include <boost/type_traits/is_arithmetic.hpp>
#include <boost/type_traits/is_enum.hpp>
#include <boost/utility/enable_if.hpp>
#include "shared_nts.hpp"

namespace Poseidon {

#ifdef POSEIDON_CXX11
using JsonNull = std::nullptr_t;
#else
typedef struct JsonNull_ *JsonNull;
#endif

class JsonElement;

class JsonObject {
public:
	typedef boost::container::map<SharedNts, JsonElement> base_container;

	typedef base_container::value_type        value_type;
	typedef base_container::const_reference   const_reference;
	typedef base_container::reference         reference;
	typedef base_container::size_type         size_type;
	typedef base_container::difference_type   difference_type;

	typedef base_container::const_iterator          const_iterator;
	typedef base_container::iterator                iterator;
	typedef base_container::const_reverse_iterator  const_reverse_iterator;
	typedef base_container::reverse_iterator        reverse_iterator;

private:
	base_container m_elements;

public:
	JsonObject();
	explicit JsonObject(std::istream &is);
#ifndef POSEIDON_CXX11
	JsonObject(const JsonObject &rhs);
	JsonObject &operator=(const JsonObject &rhs);
#endif

public:
	bool empty() const;
	size_type size() const;
	void clear();

	const_iterator begin() const;
	iterator begin();
	const_iterator cbegin() const;
	const_iterator end() const;
	iterator end();
	const_iterator cend() const;

	const_reverse_iterator rbegin() const;
	reverse_iterator rbegin();
	const_reverse_iterator crbegin() const;
	const_reverse_iterator rend() const;
	reverse_iterator rend();
	const_reverse_iterator crend() const;

	iterator erase(const_iterator pos);
	iterator erase(const_iterator first, const_iterator last);
	bool erase(const char *key);
	bool erase(const SharedNts &key);

	const_iterator find(const char *key) const;
	const_iterator find(const SharedNts &key) const;
	iterator find(const char *key);
	iterator find(const SharedNts &key);

	bool has(const char *key) const;
	bool has(const SharedNts &key);
	const JsonElement &get(const char *key) const { // 若指定的键不存在，则返回空元素。
		return get(SharedNts::view(key));
	};
	const JsonElement &get(const SharedNts &key) const;
	const JsonElement &at(const char *key) const { // 若指定的键不存在，则抛出 std::out_of_range。
		return at(SharedNts::view(key));
	};
	const JsonElement &at(const SharedNts &key) const;
	JsonElement &at(const char *key){ // 若指定的键不存在，则抛出 std::out_of_range。
		return at(SharedNts::view(key));
	};
	JsonElement &at(const SharedNts &key);
	iterator set(SharedNts key, JsonElement val);
#ifdef POSEIDON_CXX11
	template<typename KeyT, typename ...ParamsT>
	iterator emplace(KeyT &&key, ParamsT &&...params);
#endif

	void swap(JsonObject &rhs) NOEXCEPT;

	std::string dump() const;
	void dump(std::ostream &os) const;
	void parse(std::istream &is);
};

inline void swap(JsonObject &lhs, JsonObject &rhs) NOEXCEPT {
	lhs.swap(rhs);
}

class JsonArray {
public:
	typedef boost::container::deque<JsonElement> base_container;

	typedef base_container::value_type        value_type;
	typedef base_container::const_reference   const_reference;
	typedef base_container::reference         reference;
	typedef base_container::size_type         size_type;
	typedef base_container::difference_type   difference_type;

	typedef base_container::const_iterator          const_iterator;
	typedef base_container::iterator                iterator;
	typedef base_container::const_reverse_iterator  const_reverse_iterator;
	typedef base_container::reverse_iterator        reverse_iterator;

private:
	base_container m_elements;

public:
	JsonArray();
	explicit JsonArray(std::istream &is);
#ifndef POSEIDON_CXX11
	JsonArray(const JsonArray &rhs);
	JsonArray &operator=(const JsonArray &rhs);
#endif

public:
	bool empty() const;
	size_type size() const;
	void clear();

	const_iterator begin() const;
	iterator begin();
	const_iterator cbegin() const;
	const_iterator end() const;
	iterator end();
	const_iterator cend() const;

	const_reverse_iterator rbegin() const;
	reverse_iterator rbegin();
	const_reverse_iterator crbegin() const;
	const_reverse_iterator rend() const;
	reverse_iterator rend();
	const_reverse_iterator crend() const;

	iterator erase(const_iterator pos);
	iterator erase(const_iterator first, const_iterator last);
	bool erase(size_type index);

	bool has(size_type index) const;
	const JsonElement &get(size_type index) const; // 若指定的下标不存在，则返回空元素。
	const JsonElement &at(size_type index) const; // 若指定的下标不存在，则抛出 std::out_of_range。
	JsonElement &at(size_type index); // 若指定的下标不存在，则抛出 std::out_of_range。
	JsonElement &push_front(JsonElement val);
	void pop_front();
	JsonElement &push_back(JsonElement val);
	void pop_back();
	iterator insert(const_iterator pos, JsonElement val);
#ifdef POSEIDON_CXX11
	template<typename ...ParamsT>
	JsonElement &emplace_front(ParamsT &&...params);
	template<typename ...ParamsT>
	JsonElement &emplace_back(ParamsT &&...params);
	template<typename ...ParamsT>
	iterator emplace(const_iterator pos, ParamsT &&...params);
#endif

	void swap(JsonArray &rhs) NOEXCEPT;

	std::string dump() const;
	void dump(std::ostream &os) const;
	void parse(std::istream &is);
};

inline void swap(JsonArray &lhs, JsonArray &rhs) NOEXCEPT {
	lhs.swap(rhs);
}

class JsonElement {
public:
	enum Type {
		T_BOOL, T_NUMBER, T_STRING, T_OBJECT, T_ARRAY, T_NULL,
	};

public:
	static const char *get_type_string(Type type);

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
	Type get_type() const {
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
		using std::swap;
		swap(m_data, rhs.m_data);
	}

	std::string dump() const;
	void dump(std::ostream &os) const;
	void parse(std::istream &is);
};

inline void swap(JsonElement &lhs, JsonElement &rhs) NOEXCEPT {
	lhs.swap(rhs);
}

extern const JsonElement &null_json_element() NOEXCEPT;

inline JsonObject::JsonObject()
	: m_elements()
{
}
inline JsonObject::JsonObject(std::istream &is)
	: m_elements()
{
	parse(is);
}
#ifndef POSEIDON_CXX11
inline JsonObject::JsonObject(const JsonObject &rhs)
	: m_elements(rhs.m_elements)
{
}
inline JsonObject &JsonObject::operator=(const JsonObject &rhs){
	m_elements = rhs.m_elements;
	return *this;
}
#endif

inline bool JsonObject::empty() const {
	return m_elements.empty();
}
inline JsonObject::size_type JsonObject::size() const {
	return m_elements.size();
}
inline void JsonObject::clear(){
	m_elements.clear();
}

inline JsonObject::const_iterator JsonObject::begin() const {
	return m_elements.begin();
}
inline JsonObject::iterator JsonObject::begin(){
	return m_elements.begin();
}
inline JsonObject::const_iterator JsonObject::cbegin() const {
	return m_elements.begin();
}
inline JsonObject::const_iterator JsonObject::end() const {
	return m_elements.end();
}
inline JsonObject::iterator JsonObject::end(){
	return m_elements.end();
}
inline JsonObject::const_iterator JsonObject::cend() const {
	return m_elements.end();
}

inline JsonObject::const_reverse_iterator JsonObject::rbegin() const {
	return m_elements.rbegin();
}
inline JsonObject::reverse_iterator JsonObject::rbegin(){
	return m_elements.rbegin();
}
inline JsonObject::const_reverse_iterator JsonObject::crbegin() const {
	return m_elements.rbegin();
}
inline JsonObject::const_reverse_iterator JsonObject::rend() const {
	return m_elements.rend();
}
inline JsonObject::reverse_iterator JsonObject::rend(){
	return m_elements.rend();
}
inline JsonObject::const_reverse_iterator JsonObject::crend() const {
	return m_elements.rend();
}

inline JsonObject::iterator JsonObject::erase(JsonObject::const_iterator pos){
	return m_elements.erase(pos);
}
inline JsonObject::iterator JsonObject::erase(JsonObject::const_iterator first, JsonObject::const_iterator last){
	return m_elements.erase(first, last);
}
inline bool JsonObject::erase(const char *key){
	return erase(SharedNts::view(key));
}
inline bool JsonObject::erase(const SharedNts &key){
	return m_elements.erase(key);
}

inline JsonObject::const_iterator JsonObject::find(const char *key) const {
	return find(SharedNts::view(key));
}
inline JsonObject::const_iterator JsonObject::find(const SharedNts &key) const {
	return m_elements.find(key);
}
inline JsonObject::iterator JsonObject::find(const char *key){
	return find(SharedNts::view(key));
}
inline JsonObject::iterator JsonObject::find(const SharedNts &key){
	return m_elements.find(key);
}

inline bool JsonObject::has(const char *key) const {
	const AUTO(it, find(key));
	if(it == end()){
		return false;
	}
	return true;
}
inline bool JsonObject::has(const SharedNts &key){
	const AUTO(it, find(key));
	if(it == end()){
		return false;
	}
	return true;
}
inline const JsonElement &JsonObject::get(const SharedNts &key) const {
	const AUTO(it, find(key));
	if(it == end()){
		return null_json_element();
	}
	return it->second;
}
inline const JsonElement &JsonObject::at(const SharedNts &key) const {
	const AUTO(it, find(key));
	if(it == end()){
		throw std::out_of_range(__PRETTY_FUNCTION__);
	}
	return it->second;
}
inline JsonElement &JsonObject::at(const SharedNts &key){
	const AUTO(it, find(key));
	if(it == end()){
		throw std::out_of_range(__PRETTY_FUNCTION__);
	}
	return it->second;
}
inline JsonObject::iterator JsonObject::set(SharedNts key, JsonElement val){
	const AUTO(existent, m_elements.equal_range(key));
	const AUTO(hint, m_elements.erase(existent.first, existent.second));
	return m_elements.insert(hint, std::make_pair(STD_MOVE_IDN(key), STD_MOVE_IDN(val)));
}
#ifdef POSEIDON_CXX11
template<typename KeyT, typename ...ParamsT>
inline JsonObject::iterator JsonObject::emplace(KeyT &&key, ParamsT &&...params){
	const AUTO(existent, m_elements.equal_range(key));
	const AUTO(hint, m_elements.erase(existent.first, existent.second));
	return m_elements.emplace_hint(hint, std::forward<KeyT>(key), std::forward<ParamsT>(params)...);
}
#endif

inline void JsonObject::swap(JsonObject &rhs) NOEXCEPT {
	using std::swap;
	swap(m_elements, rhs.m_elements);
}

inline JsonArray::JsonArray()
	: m_elements()
{
}
inline JsonArray::JsonArray(std::istream &is)
	: m_elements()
{
	parse(is);
}
#ifndef POSEIDON_CXX11
inline JsonArray::JsonArray(const JsonArray &rhs)
	: m_elements(rhs.m_elements)
{
}
inline JsonArray &JsonArray::operator=(const JsonArray &rhs){
	m_elements = rhs.m_elements;
	return *this;
}
#endif

inline bool JsonArray::empty() const {
	return m_elements.empty();
}
inline JsonArray::size_type JsonArray::size() const {
	return m_elements.size();
}
inline void JsonArray::clear(){
	m_elements.clear();
}

inline JsonArray::const_iterator JsonArray::begin() const {
	return m_elements.begin();
}
inline JsonArray::iterator JsonArray::begin(){
	return m_elements.begin();
}
inline JsonArray::const_iterator JsonArray::cbegin() const {
	return m_elements.begin();
}
inline JsonArray::const_iterator JsonArray::end() const {
	return m_elements.end();
}
inline JsonArray::iterator JsonArray::end(){
	return m_elements.end();
}
inline JsonArray::const_iterator JsonArray::cend() const {
	return m_elements.end();
}

inline JsonArray::const_reverse_iterator JsonArray::rbegin() const {
	return m_elements.rbegin();
}
inline JsonArray::reverse_iterator JsonArray::rbegin(){
	return m_elements.rbegin();
}
inline JsonArray::const_reverse_iterator JsonArray::crbegin() const {
	return m_elements.rbegin();
}
inline JsonArray::const_reverse_iterator JsonArray::rend() const {
	return m_elements.rend();
}
inline JsonArray::reverse_iterator JsonArray::rend(){
	return m_elements.rend();
}
inline JsonArray::const_reverse_iterator JsonArray::crend() const {
	return m_elements.rend();
}

inline JsonArray::iterator JsonArray::erase(JsonArray::const_iterator pos){
	return m_elements.erase(pos);
}
inline JsonArray::iterator JsonArray::erase(JsonArray::const_iterator first, JsonArray::const_iterator last){
	return m_elements.erase(first, last);
}
inline bool JsonArray::erase(JsonArray::size_type index){
	if(index >= size()){
		return false;
	}
	m_elements.erase(begin() + static_cast<difference_type>(index));
	return true;
}

inline bool JsonArray::has(JsonArray::size_type index) const {
	if(index >= size()){
		return false;
	}
	return true;
}
inline const JsonElement &JsonArray::get(JsonArray::size_type index) const {
	if(index >= size()){
		return null_json_element();
	}
	return begin()[static_cast<difference_type>(index)];
}
inline const JsonElement &JsonArray::at(JsonArray::size_type index) const {
	if(index >= size()){
		throw std::out_of_range(__PRETTY_FUNCTION__);
	}
	return begin()[static_cast<difference_type>(index)];
}
inline JsonElement &JsonArray::at(JsonArray::size_type index){
	if(index >= size()){
		throw std::out_of_range(__PRETTY_FUNCTION__);
	}
	return begin()[static_cast<difference_type>(index)];
}
inline JsonElement &JsonArray::push_front(JsonElement val){
	m_elements.push_front(STD_MOVE(val));
	return m_elements.front();
}
inline void JsonArray::pop_front(){
	m_elements.pop_front();
}
inline JsonElement &JsonArray::push_back(JsonElement val){
	m_elements.push_back(STD_MOVE(val));
	return m_elements.front();
}
inline void JsonArray::pop_back(){
	m_elements.pop_back();
}
inline JsonArray::iterator JsonArray::insert(JsonArray::const_iterator pos, JsonElement val){
	return m_elements.insert(pos, STD_MOVE_IDN(val));
}
#ifdef POSEIDON_CXX11
template<typename ...ParamsT>
inline JsonElement &JsonArray::emplace_front(ParamsT &&...params){
	m_elements.emplace_front(std::forward<ParamsT>(params)...);
	return m_elements.front();
}
template<typename ...ParamsT>
inline JsonElement &JsonArray::emplace_back(ParamsT &&...params){
	m_elements.emplace_back(std::forward<ParamsT>(params)...);
	return m_elements.back();
}
template<typename ...ParamsT>
inline JsonArray::iterator JsonArray::emplace(const_iterator pos, ParamsT &&...params){
	return m_elements.insert(pos, std::forward<ParamsT>(params)...);
}
#endif

inline void JsonArray::swap(JsonArray &rhs) NOEXCEPT {
	using std::swap;
	swap(m_elements, rhs.m_elements);
}

inline std::ostream &operator<<(std::ostream &os, const JsonElement &rhs){
	rhs.dump(os);
	return os;
}
inline std::istream &operator>>(std::istream &is, JsonElement &rhs){
	rhs.parse(is);
	return is;
}

inline std::ostream &operator<<(std::ostream &os, const JsonArray &rhs){
	rhs.dump(os);
	return os;
}
inline std::istream &operator>>(std::istream &is, JsonArray &rhs){
	rhs.parse(is);
	return is;
}

inline std::ostream &operator<<(std::ostream &os, const JsonObject &rhs){
	rhs.dump(os);
	return os;
}
inline std::istream &operator>>(std::istream &is, JsonObject &rhs){
	rhs.parse(is);
	return is;
}

}

#endif
