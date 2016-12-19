// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_MULTIPART_HPP_
#define POSEIDON_HTTP_MULTIPART_HPP_

#include "../cxx_ver.hpp"
#include <boost/container/deque.hpp>
#include <stdexcept>
#include "../optional_map.hpp"
#include "../stream_buffer.hpp"

namespace Poseidon {

namespace Http {
	struct MultipartElement {
		OptionalMap headers;
		StreamBuffer entity;
	};

	extern const MultipartElement &empty_multipart_element() NOEXCEPT;

	class Multipart {
	public:
		typedef boost::container::deque<MultipartElement> base_container;

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
		std::string m_boundary;
		base_container m_elements;

	public:
		Multipart()
			: m_boundary(), m_elements()
		{
		}
		Multipart(std::string boundary, std::istream &is)
			: m_boundary(STD_MOVE(boundary)), m_elements()
		{
			parse(is);
		}
#ifndef POSEIDON_CXX11
		Multipart(const Multipart &rhs)
			: m_boundary(rhs.m_boundary), m_elements(rhs.m_elements)
		{
		}
		Multipart &operator=(const Multipart &rhs){
			m_boundary = rhs.m_boundary;
			m_elements = rhs.m_elements;
			return *this;
		}
#endif

	public:
		const std::string &get_boundary() const {
			return m_boundary;
		}
		void set_boundary(std::string boundary){
			m_boundary.swap(boundary);
		}
		void random_boundary();

		bool empty() const {
			return m_elements.empty();
		}
		size_type size() const {
			return m_elements.size();
		}
		void clear(){
			m_elements.clear();
		}

		const_iterator begin() const {
			return m_elements.begin();
		}
		iterator begin(){
			return m_elements.begin();
		}
		const_iterator cbegin() const {
			return m_elements.begin();
		}
		const_iterator end() const {
			return m_elements.end();
		}
		iterator end(){
			return m_elements.end();
		}
		const_iterator cend() const {
			return m_elements.end();
		}

		const_reverse_iterator rbegin() const {
			return m_elements.rbegin();
		}
		reverse_iterator rbegin(){
			return m_elements.rbegin();
		}
		const_reverse_iterator crbegin() const {
			return m_elements.rbegin();
		}
		const_reverse_iterator rend() const {
			return m_elements.rend();
		}
		reverse_iterator rend(){
			return m_elements.rend();
		}
		const_reverse_iterator crend() const {
			return m_elements.rend();
		}

		iterator erase(const_iterator pos){
			return m_elements.erase(pos);
		}
		iterator erase(const_iterator first, const_iterator last){
			return m_elements.erase(first, last);
		}
		bool erase(size_type index){
			if(index >= size()){
				return false;
			}
			m_elements.erase(begin() + static_cast<difference_type>(index));
			return true;
		}

		bool has(size_type index) const {
			if(index >= size()){
				return false;
			}
			return true;
		}
		const MultipartElement &get(size_type index) const { // 若指定的下标不存在，则返回空元素。
			if(index >= size()){
				return empty_multipart_element();
			}
			return begin()[static_cast<difference_type>(index)];
		}
		const MultipartElement &at(size_type index) const { // 若指定的下标不存在，则抛出 std::out_of_range。
			if(index >= size()){
				throw std::out_of_range(__PRETTY_FUNCTION__);
			}
			return begin()[static_cast<difference_type>(index)];
		}
		MultipartElement &at(size_type index){ // 若指定的下标不存在，则抛出 std::out_of_range。
			if(index >= size()){
				throw std::out_of_range(__PRETTY_FUNCTION__);
			}
			return begin()[static_cast<difference_type>(index)];
		}
		MultipartElement &push_front(MultipartElement val){
			m_elements.push_front(STD_MOVE(val));
			return m_elements.front();
		}
		void pop_front(){
			m_elements.pop_front();
		}
		MultipartElement &push_back(MultipartElement val){
			m_elements.push_back(STD_MOVE(val));
			return m_elements.back();
		}
		void pop_back(){
			m_elements.pop_back();
		}
		iterator insert(const_iterator pos, MultipartElement val){
			return m_elements.insert(pos, STD_MOVE(val));
		}
#ifdef POSEIDON_CXX11
		template<typename ...ParamsT>
		MultipartElement &emplace_front(ParamsT &&...params){
			m_elements.emplace_front(std::forward<ParamsT>(params)...);
			return m_elements.front();
		}
		template<typename ...ParamsT>
		MultipartElement &emplace_back(ParamsT &&...params){
			m_elements.emplace_back(std::forward<ParamsT>(params)...);
			return m_elements.back();
		}
		template<typename ...ParamsT>
		iterator emplace(const_iterator pos, ParamsT &&...params){
			return m_elements.emplace(pos, std::forward<ParamsT>(params)...);
		}
#endif

		void swap(Multipart &rhs) NOEXCEPT {
			using std::swap;
			swap(m_elements, rhs.m_elements);
		}

		std::string dump() const;
		void dump(std::ostream &os) const;
		void parse(std::istream &is);
	};

	inline void swap(Multipart &lhs, Multipart &rhs) NOEXCEPT {
		lhs.swap(rhs);
	}

	inline std::ostream &operator<<(std::ostream &os, const Multipart &rhs){
		rhs.dump(os);
		return os;
	}
	inline std::istream &operator>>(std::istream &is, Multipart &rhs){
		rhs.parse(is);
		return is;
	}
}

}

#endif
