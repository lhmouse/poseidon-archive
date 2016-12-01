// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_FORM_DATA_HPP_
#define POSEIDON_HTTP_FORM_DATA_HPP_

#include "../cxx_ver.hpp"
#include "header_option.hpp"
#include <boost/container/map.hpp>
#include <stdexcept>
#include "../shared_nts.hpp"
#include "../optional_map.hpp"
#include "../stream_buffer.hpp"

namespace Poseidon {

namespace Http {
	struct FormDataElement {
		std::string filename;
		OptionalMap headers;
		StreamBuffer entity;
	};

	extern const FormDataElement &empty_form_data_element() NOEXCEPT;

	class FormData {
	public:
		typedef boost::container::map<SharedNts, FormDataElement> base_container;

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
		FormData()
			: m_boundary(), m_elements()
		{
		}
		FormData(std::string boundary, std::istream &is)
			: m_boundary(STD_MOVE(boundary)), m_elements()
		{
			parse(is);
		}
#ifndef POSEIDON_CXX11
		FormData(const FormData &rhs)
			: m_boundary(rhs.m_boundary), m_elements(rhs.m_elements)
		{
		}
		FormData &operator=(const FormData &rhs){
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
		size_type erase(const char *key){
			return erase(SharedNts::view(key));
		}
		size_type erase(const SharedNts &key){
			return m_elements.erase(key);
		}

		const_iterator find(const char *key) const {
			return find(SharedNts::view(key));
		}
		const_iterator find(const SharedNts &key) const {
			return m_elements.find(key);
		}
		iterator find(const char *key){
			return find(SharedNts::view(key));
		}
		iterator find(const SharedNts &key){
			return m_elements.find(key);
		}

		bool has(const char *key) const {
			return find(key) != end();
		}
		bool has(const SharedNts &key){
			return find(key) != end();
		}
		iterator set(SharedNts key, FormDataElement val){
			const AUTO(existent, m_elements.equal_range(key));
			const AUTO(hint, m_elements.erase(existent.first, existent.second));
			return m_elements.insert(hint, std::make_pair(STD_MOVE_IDN(key), STD_MOVE_IDN(val)));
		}

		const FormDataElement &get(const char *key) const { // 若指定的键不存在，则返回空的 FormDataElement。
			return get(SharedNts::view(key));
		};
		const FormDataElement &get(const SharedNts &key) const {
			const AUTO(it, find(key));
			if(it == end()){
				return empty_form_data_element();
			}
			return it->second;
		}
		const FormDataElement &at(const char *key) const { // 若指定的键不存在，则抛出 std::out_of_range。
			return at(SharedNts::view(key));
		};
		const FormDataElement &at(const SharedNts &key) const {
			const AUTO(it, find(key));
			if(it == end()){
				throw std::out_of_range(__PRETTY_FUNCTION__);
			}
			return it->second;
		}

		void swap(FormData &rhs) NOEXCEPT {
			using std::swap;
			swap(m_elements, rhs.m_elements);
		}

		std::string dump() const;
		void dump(std::ostream &os) const;
		void parse(std::istream &is);
	};

	inline void swap(FormData &lhs, FormData &rhs) NOEXCEPT {
		lhs.swap(rhs);
	}

	inline std::ostream &operator<<(std::ostream &os, const FormData &rhs){
		rhs.dump(os);
		return os;
	}
	inline std::istream &operator>>(std::istream &is, FormData &rhs){
		rhs.parse(is);
		return is;
	}
}

}

#endif
