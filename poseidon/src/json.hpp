// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

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
#include <boost/utility/enable_if.hpp>
#include "rcnts.hpp"

namespace Poseidon {

class Stream_buffer;
class Json_object;
class Json_array;
class Json_element;

#ifdef POSEIDON_CXX11
using Json_null = std::nullptr_t;
#else
typedef struct Json_null_ *Json_null;
#endif

extern const Json_element & null_json_element() NOEXCEPT;

class Json_object {
public:
	typedef boost::container::map<Rcnts, Json_element> base_container;

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
	Json_object();
	explicit Json_object(std::istream &is);
#ifndef POSEIDON_CXX11
	Json_object(const Json_object &rhs);
	Json_object & operator=(const Json_object &rhs);
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
	bool erase(const Rcnts &key);

	const_iterator find(const char *key) const;
	const_iterator find(const Rcnts &key) const;
	iterator find(const char *key);
	iterator find(const Rcnts &key);

	bool has(const char *key) const;
	bool has(const Rcnts &key) const;
	const Json_element & get(const Rcnts &key) const; // 若指定的键不存在，则返回空元素。
	const Json_element & at(const Rcnts &key) const; // 若指定的键不存在，则抛出 std::out_of_range。
	Json_element & at(const Rcnts &key); // 若指定的键不存在，则抛出 std::out_of_range。
	const Json_element & get(const char *key) const {
		return get(Rcnts::view(key));
	}
	const Json_element & at(const char *key) const {
		return at(Rcnts::view(key));
	}
	Json_element & at(const char *key){
		return at(Rcnts::view(key));
	}
	template<typename T>
	const T & get(const Rcnts &key) const;
	template<typename T>
	const T & at(const Rcnts &key) const;
	template<typename T>
	T & at(const Rcnts &key);
	template<typename T>
	const T & get(const char *key) const {
		return get<T>(Rcnts::view(key));
	}
	template<typename T>
	const T & at(const char *key) const {
		return at<T>(Rcnts::view(key));
	}
	template<typename T>
	T & at(const char *key){
		return at<T>(Rcnts::view(key));
	}
	iterator set(Rcnts key, Json_element val);
#ifdef POSEIDON_CXX11
	template<typename KeyT, typename ...ParamsT>
	iterator emplace_or_assign(KeyT &&key, ParamsT &&...params);
#endif

	void swap(Json_object &rhs) NOEXCEPT;

	Stream_buffer dump() const;
	void dump(std::ostream &os) const;
	void parse(std::istream &is);
};

inline void swap(Json_object &lhs, Json_object &rhs) NOEXCEPT {
	lhs.swap(rhs);
}

inline std::ostream & operator<<(std::ostream &os, const Json_object &rhs){
	rhs.dump(os);
	return os;
}
inline std::istream & operator>>(std::istream &is, Json_object &rhs){
	rhs.parse(is);
	return is;
}

class Json_array {
public:
	typedef boost::container::deque<Json_element> base_container;

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
	Json_array();
	explicit Json_array(std::istream &is);
#ifndef POSEIDON_CXX11
	Json_array(const Json_array &rhs);
	Json_array & operator=(const Json_array &rhs);
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
	const Json_element & get(size_type index) const; // 若指定的下标不存在，则返回空元素。
	const Json_element & at(size_type index) const; // 若指定的下标不存在，则抛出 std::out_of_range。
	Json_element & at(size_type index); // 若指定的下标不存在，则抛出 std::out_of_range。
	template<typename T>
	const T & get(size_type index) const;
	template<typename T>
	const T & at(size_type index) const;
	template<typename T>
	T & at(size_type index);
	Json_element & push_front(Json_element val);
	void pop_front();
	Json_element & push_back(Json_element val);
	void pop_back();
	iterator insert(const_iterator pos, Json_element val);
#ifdef POSEIDON_CXX11
	template<typename ...ParamsT>
	Json_element & emplace_front(ParamsT &&...params);
	template<typename ...ParamsT>
	Json_element & emplace_back(ParamsT &&...params);
	template<typename ...ParamsT>
	iterator emplace(const_iterator pos, ParamsT &&...params);
#endif

	void swap(Json_array &rhs) NOEXCEPT;

	Stream_buffer dump() const;
	void dump(std::ostream &os) const;
	void parse(std::istream &is);
};

inline void swap(Json_array &lhs, Json_array &rhs) NOEXCEPT {
	lhs.swap(rhs);
}

inline std::ostream & operator<<(std::ostream &os, const Json_array &rhs){
	rhs.dump(os);
	return os;
}
inline std::istream & operator>>(std::istream &is, Json_array &rhs){
	rhs.parse(is);
	return is;
}

class Json_element {
public:
	enum Type {
		type_null    = 0,
		type_boolean = 1,
		type_number  = 2,
		type_string  = 3,
		type_object  = 4,
		type_array   = 5,
	};

	typedef Json_null    Type_null;
	typedef bool        Type_boolean;
	typedef double      Type_number;
	typedef std::string Type_string;
	typedef Json_object  Type_object;
	typedef Json_array   Type_array;

public:
	static const char * get_type_string(Type type);

private:
	boost::variant< Type_null
	              , Type_boolean
	              , Type_number
	              , Type_string
	              , Type_object
	              , Type_array
		> m_variant;

public:
	Json_element(Json_null = Json_null())
		: m_variant(Json_null())
	{
		//
	}
	Json_element(bool rhs)
		: m_variant(rhs)
	{
		//
	}
#ifdef POSEIDON_CXX11
	template<typename T, typename std::enable_if<std::is_arithmetic<T>::value || std::is_enum<T>::value>::type * = nullptr>
	Json_element(T rhs)
#else
	template<typename T>
	Json_element(T rhs, typename std::enable_if<std::is_arithmetic<T>::value || std::is_enum<T>::value>::type * = 0)
#endif
		: m_variant(static_cast<double>(rhs))
	{
		//
	}
	Json_element(const char *rhs)
		: m_variant(std::string(rhs))
	{
		//
	}
	Json_element(std::string rhs)
		: m_variant(STD_MOVE_IDN(rhs))
	{
		//
	}
	Json_element(Json_object rhs)
		: m_variant(STD_MOVE_IDN(rhs))
	{
		//
	}
	Json_element(Json_array rhs)
		: m_variant(STD_MOVE_IDN(rhs))
	{
		//
	}

public:
	Type get_type() const {
		return static_cast<Type>(m_variant.which());
	}

	template<typename T>
	const T & get() const {
		return boost::get<T>(m_variant);
	}
	template<typename T>
	T & get(){
		return boost::get<T>(m_variant);
	}
	template<typename T>
	void set(T rhs){
		Json_element(STD_MOVE_IDN(rhs)).swap(*this);
	}

	void swap(Json_element &rhs) NOEXCEPT {
		using std::swap;
		swap(m_variant, rhs.m_variant);
	}

	Stream_buffer dump() const;
	void dump(std::ostream &os) const;
	void parse(std::istream &is);
};

inline void swap(Json_element &lhs, Json_element &rhs) NOEXCEPT {
	lhs.swap(rhs);
}

inline Json_object::Json_object()
	: m_elements()
{
	//
}
#ifndef POSEIDON_CXX11
inline Json_object::Json_object(const Json_object &rhs)
	: m_elements(rhs.m_elements)
{
	//
}
inline Json_object & Json_object::operator=(const Json_object &rhs){
	m_elements = rhs.m_elements;
	return *this;
}
#endif

inline bool Json_object::empty() const {
	return m_elements.empty();
}
inline Json_object::size_type Json_object::size() const {
	return m_elements.size();
}
inline void Json_object::clear(){
	m_elements.clear();
}

inline Json_object::const_iterator Json_object::begin() const {
	return m_elements.begin();
}
inline Json_object::iterator Json_object::begin(){
	return m_elements.begin();
}
inline Json_object::const_iterator Json_object::cbegin() const {
	return m_elements.begin();
}
inline Json_object::const_iterator Json_object::end() const {
	return m_elements.end();
}
inline Json_object::iterator Json_object::end(){
	return m_elements.end();
}
inline Json_object::const_iterator Json_object::cend() const {
	return m_elements.end();
}

inline Json_object::const_reverse_iterator Json_object::rbegin() const {
	return m_elements.rbegin();
}
inline Json_object::reverse_iterator Json_object::rbegin(){
	return m_elements.rbegin();
}
inline Json_object::const_reverse_iterator Json_object::crbegin() const {
	return m_elements.rbegin();
}
inline Json_object::const_reverse_iterator Json_object::rend() const {
	return m_elements.rend();
}
inline Json_object::reverse_iterator Json_object::rend(){
	return m_elements.rend();
}
inline Json_object::const_reverse_iterator Json_object::crend() const {
	return m_elements.rend();
}

inline Json_object::iterator Json_object::erase(Json_object::const_iterator pos){
	return m_elements.erase(pos);
}
inline Json_object::iterator Json_object::erase(Json_object::const_iterator first, Json_object::const_iterator last){
	return m_elements.erase(first, last);
}
inline bool Json_object::erase(const char *key){
	return erase(Rcnts::view(key));
}
inline bool Json_object::erase(const Rcnts &key){
	return m_elements.erase(key);
}

inline Json_object::const_iterator Json_object::find(const char *key) const {
	return find(Rcnts::view(key));
}
inline Json_object::const_iterator Json_object::find(const Rcnts &key) const {
	return m_elements.find(key);
}
inline Json_object::iterator Json_object::find(const char *key){
	return find(Rcnts::view(key));
}
inline Json_object::iterator Json_object::find(const Rcnts &key){
	return m_elements.find(key);
}

inline bool Json_object::has(const char *key) const {
	const AUTO(it, find(key));
	if(it == end()){
		return false;
	}
	return true;
}
inline bool Json_object::has(const Rcnts &key) const {
	const AUTO(it, find(key));
	if(it == end()){
		return false;
	}
	return true;
}
inline const Json_element & Json_object::get(const Rcnts &key) const {
	const AUTO(it, find(key));
	if(it == end()){
		return null_json_element();
	}
	return it->second;
}
inline const Json_element & Json_object::at(const Rcnts &key) const {
	const AUTO(it, find(key));
	if(it == end()){
		throw std::out_of_range(__PRETTY_FUNCTION__);
	}
	return it->second;
}
inline Json_element & Json_object::at(const Rcnts &key){
	const AUTO(it, find(key));
	if(it == end()){
		throw std::out_of_range(__PRETTY_FUNCTION__);
	}
	return it->second;
}
template<typename T>
const T & Json_object::get(const Rcnts &key) const {
	return get(key).get<T>();
}
template<typename T>
const T & Json_object::at(const Rcnts &key) const {
	return at(key).get<T>();
}
template<typename T>
T & Json_object::at(const Rcnts &key){
	return at(key).get<T>();
}
inline Json_object::iterator Json_object::set(Rcnts key, Json_element val){
	AUTO(it, m_elements.find(key));
	if(it == m_elements.end()){
		it = m_elements.emplace(STD_MOVE_IDN(key), STD_MOVE_IDN(val)).first;
	} else {
		it->second.swap(val);
	}
	return it;
}
#ifdef POSEIDON_CXX11
template<typename KeyT, typename ...ParamsT>
inline Json_object::iterator Json_object::emplace_or_assign(KeyT &&key, ParamsT &&...params){
	AUTO(it, m_elements.find(key));
	if(it == m_elements.end()){
		it = m_elements.emplace(std::forward<KeyT>(key), std::forward<ParamsT>(params)...).first;
	} else {
		it->second = Json_element(std::forward<ParamsT>(params)...);
	}
	return it;
}
#endif

inline void Json_object::swap(Json_object &rhs) NOEXCEPT {
	using std::swap;
	swap(m_elements, rhs.m_elements);
}

inline Json_array::Json_array()
	: m_elements()
{
	//
}
#ifndef POSEIDON_CXX11
inline Json_array::Json_array(const Json_array &rhs)
	: m_elements(rhs.m_elements)
{
	//
}
inline Json_array & Json_array::operator=(const Json_array &rhs){
	m_elements = rhs.m_elements;
	return *this;
}
#endif

inline bool Json_array::empty() const {
	return m_elements.empty();
}
inline Json_array::size_type Json_array::size() const {
	return m_elements.size();
}
inline void Json_array::clear(){
	m_elements.clear();
}

inline Json_array::const_iterator Json_array::begin() const {
	return m_elements.begin();
}
inline Json_array::iterator Json_array::begin(){
	return m_elements.begin();
}
inline Json_array::const_iterator Json_array::cbegin() const {
	return m_elements.begin();
}
inline Json_array::const_iterator Json_array::end() const {
	return m_elements.end();
}
inline Json_array::iterator Json_array::end(){
	return m_elements.end();
}
inline Json_array::const_iterator Json_array::cend() const {
	return m_elements.end();
}

inline Json_array::const_reverse_iterator Json_array::rbegin() const {
	return m_elements.rbegin();
}
inline Json_array::reverse_iterator Json_array::rbegin(){
	return m_elements.rbegin();
}
inline Json_array::const_reverse_iterator Json_array::crbegin() const {
	return m_elements.rbegin();
}
inline Json_array::const_reverse_iterator Json_array::rend() const {
	return m_elements.rend();
}
inline Json_array::reverse_iterator Json_array::rend(){
	return m_elements.rend();
}
inline Json_array::const_reverse_iterator Json_array::crend() const {
	return m_elements.rend();
}

inline Json_array::iterator Json_array::erase(Json_array::const_iterator pos){
	return m_elements.erase(pos);
}
inline Json_array::iterator Json_array::erase(Json_array::const_iterator first, Json_array::const_iterator last){
	return m_elements.erase(first, last);
}
inline bool Json_array::erase(Json_array::size_type index){
	if(index >= size()){
		return false;
	}
	m_elements.erase(begin() + static_cast<difference_type>(index));
	return true;
}

inline bool Json_array::has(Json_array::size_type index) const {
	if(index >= size()){
		return false;
	}
	return true;
}
inline const Json_element & Json_array::get(Json_array::size_type index) const {
	if(index >= size()){
		return null_json_element();
	}
	return begin()[static_cast<difference_type>(index)];
}
inline const Json_element & Json_array::at(Json_array::size_type index) const {
	if(index >= size()){
		throw std::out_of_range(__PRETTY_FUNCTION__);
	}
	return begin()[static_cast<difference_type>(index)];
}
inline Json_element & Json_array::at(Json_array::size_type index){
	if(index >= size()){
		throw std::out_of_range(__PRETTY_FUNCTION__);
	}
	return begin()[static_cast<difference_type>(index)];
}
template<typename T>
inline const T & Json_array::get(Json_array::size_type index) const {
	return get(index).get<T>();
}
template<typename T>
inline const T & Json_array::at(Json_array::size_type index) const {
	return at(index).get<T>();
}
template<typename T>
inline T & Json_array::at(Json_array::size_type index){
	return at(index).get<T>();
}
inline Json_element & Json_array::push_front(Json_element val){
	m_elements.push_front(STD_MOVE(val));
	return m_elements.front();
}
inline void Json_array::pop_front(){
	m_elements.pop_front();
}
inline Json_element & Json_array::push_back(Json_element val){
	m_elements.push_back(STD_MOVE(val));
	return m_elements.front();
}
inline void Json_array::pop_back(){
	m_elements.pop_back();
}
inline Json_array::iterator Json_array::insert(Json_array::const_iterator pos, Json_element val){
	return m_elements.insert(pos, STD_MOVE_IDN(val));
}
#ifdef POSEIDON_CXX11
template<typename ...ParamsT>
inline Json_element & Json_array::emplace_front(ParamsT &&...params){
	m_elements.emplace_front(std::forward<ParamsT>(params)...);
	return m_elements.front();
}
template<typename ...ParamsT>
inline Json_element & Json_array::emplace_back(ParamsT &&...params){
	m_elements.emplace_back(std::forward<ParamsT>(params)...);
	return m_elements.back();
}
template<typename ...ParamsT>
inline Json_array::iterator Json_array::emplace(const_iterator pos, ParamsT &&...params){
	return m_elements.insert(pos, std::forward<ParamsT>(params)...);
}
#endif

inline void Json_array::swap(Json_array &rhs) NOEXCEPT {
	using std::swap;
	swap(m_elements, rhs.m_elements);
}

inline std::ostream & operator<<(std::ostream &os, const Json_element &rhs){
	rhs.dump(os);
	return os;
}
inline std::istream & operator>>(std::istream &is, Json_element &rhs){
	rhs.parse(is);
	return is;
}

}

#endif
