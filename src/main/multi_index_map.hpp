// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_MULTI_INDEX_MAP_HPP_
#define POSEIDON_MULTI_INDEX_MAP_HPP_

/*

定义时需要指定索引。

typedef ::std::pair<int, ::std::string> Item;

MULTI_INDEX_MAP(Container, Item,
	UNIQUE_MEMBER_INDEX(first)
	MULTI_MEMBER_INDEX(second)
	SEQUENCED_INDEX()
);

基本用法和 ::std::map 类似，只是 find, count, lowerBound, upperBound, equalRange 等
成员函数需要带一个非类型模板参数，指定使用第几个索引。

Container c;
c.insert(Item(1, "abc"));
c.insert(Item(2, "def"));
::std::cout <<c.find<0>(1)->second <<::std::endl;	// "abc";
assert(c.upperBound<1>("zzz") == c.end<1>());	// 通过。

*/

#include "cxx_ver.hpp"
#include "cxx_util.hpp"
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/identity.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/sequenced_index.hpp>

#define MULTI_INDEX_MAP(class_name_, value_type_, indices_)	\
	class class_name_ {	\
	public:	\
		typedef value_type_ value_type;	\
		\
		typedef ::boost::multi_index::multi_index_container<	\
			value_type, ::boost::multi_index::indexed_by<	\
				STRIP_FIRST(void indices_)>	\
			> delegated_container;	\
		\
		typedef delegated_container::const_iterator const_iterator;	\
		typedef delegated_container::iterator iterator;	\
		typedef delegated_container::const_reverse_iterator const_reverse_iterator;	\
		typedef delegated_container::reverse_iterator reverse_iterator;	\
	private:	\
		template<unsigned IndexIdT>	\
		struct KeyModifier {	\
			typedef typename delegated_container::nth_index<IndexIdT>::type::key_type key_type;	\
			\
			key_type &m_key;	\
			\
			explicit KeyModifier(key_type &key)	\
				: m_key(key)	\
			{	\
			}	\
			\
			void operator()(key_type &old){	\
				using ::std::swap;	\
				swap(old, m_key);	\
			}	\
		};	\
		\
	private:	\
		delegated_container m_delegate;	\
		\
	public:	\
		template<unsigned IndexIdT>  \
		const typename delegated_container::nth_index<IndexIdT>::type &getIndex() const {	\
			return m_delegate.get<IndexIdT>();	\
		}	\
		template<unsigned IndexIdT>  \
		typename delegated_container::nth_index<IndexIdT>::type &getIndex(){	\
			return m_delegate.get<IndexIdT>();	\
		}	\
		\
		const_iterator begin() const {	\
			return m_delegate.begin();	\
		}	\
		iterator begin(){	\
			return m_delegate.begin();	\
		}	\
		const_iterator end() const {	\
			return m_delegate.end();	\
		}	\
		iterator end(){	\
			return m_delegate.end();	\
		}	\
		\
		const_reverse_iterator rbegin() const {	\
			return m_delegate.rbegin();	\
		}	\
		reverse_iterator rbegin(){	\
			return m_delegate.rbegin();	\
		}	\
		const_reverse_iterator rend() const {	\
			return m_delegate.rend();	\
		}	\
		reverse_iterator rend(){	\
			return m_delegate.rend();	\
		}	\
		\
		template<unsigned IndexIdT>  \
		typename delegated_container::nth_index<IndexIdT>::type::const_iterator begin() const {	\
			return getIndex<IndexIdT>().begin();	\
		}	\
		template<unsigned IndexIdT>  \
		typename delegated_container::nth_index<IndexIdT>::type::iterator begin(){	\
			return getIndex<IndexIdT>().begin();	\
		}	\
		template<unsigned IndexIdT>  \
		typename delegated_container::nth_index<IndexIdT>::type::const_iterator end() const {	\
			return getIndex<IndexIdT>().end();	\
		}	\
		template<unsigned IndexIdT>  \
		typename delegated_container::nth_index<IndexIdT>::type::iterator end(){	\
			return getIndex<IndexIdT>().end();	\
		}	\
		\
		template<unsigned IndexIdT>  \
		typename delegated_container::nth_index<IndexIdT>::type::const_reverse_iterator rbegin() const {	\
			return getIndex<IndexIdT>().rbegin();	\
		}	\
		template<unsigned IndexIdT>  \
		typename delegated_container::nth_index<IndexIdT>::type::reverse_iterator rbegin(){	\
			return getIndex<IndexIdT>().rbegin();	\
		}	\
		template<unsigned IndexIdT>  \
		typename delegated_container::nth_index<IndexIdT>::type::const_reverse_iterator rend() const {	\
			return getIndex<IndexIdT>().rend();	\
		}	\
		template<unsigned IndexIdT>  \
		typename delegated_container::nth_index<IndexIdT>::type::reverse_iterator rend(){	\
			return getIndex<IndexIdT>().rend();	\
		}	\
		\
		bool empty() const {	\
			return m_delegate.empty();	\
		}	\
		::std::size_t size() const {	\
			return m_delegate.size();	\
		}	\
		void clear() {	\
			m_delegate.clear();	\
		}	\
		\
		void swap(class_name_ &rhs) throw() {	\
			m_delegate.swap(rhs.m_delegate);	\
		}	\
		\
		::std::pair<iterator, bool> insert(const value_type &val){	\
			return m_delegate.insert(val);	\
		}	\
		::std::pair<iterator, bool> insert(::Poseidon::Move<value_type> val){	\
			return m_delegate.insert(STD_MOVE(val));	\
		}	\
		iterator insert(iterator pos, const value_type &val){	\
			return m_delegate.insert(pos, val);	\
		}	\
		iterator insert(iterator pos, ::Poseidon::Move<value_type> val){	\
			return m_delegate.insert(pos, STD_MOVE(val));	\
		}	\
		\
		iterator erase(iterator pos){	\
			return m_delegate.erase(pos);	\
		}	\
		iterator erase(iterator begin, iterator end){	\
			return m_delegate.erase(begin, end);	\
		}	\
		\
		template<unsigned IndexIdT>	\
		::std::pair<typename delegated_container::nth_index<IndexIdT>::type::iterator, bool>	\
			insert(const value_type &val)	\
		{	\
			return getIndex<IndexIdT>().insert(val);	\
		}	\
		template<unsigned IndexIdT>	\
		::std::pair<typename delegated_container::nth_index<IndexIdT>::type::iterator, bool>	\
			insert(::Poseidon::Move<value_type> val)	\
		{	\
			return getIndex<IndexIdT>().insert(STD_MOVE(val));	\
		}	\
		template<unsigned IndexIdT>	\
		typename delegated_container::nth_index<IndexIdT>::type::iterator	\
			insert(typename delegated_container::nth_index<IndexIdT>::type::iterator hint,	\
				const value_type &val)	\
		{	\
			return getIndex<IndexIdT>().insert(hint, val);	\
		}	\
		template<unsigned IndexIdT>	\
		typename delegated_container::nth_index<IndexIdT>::type::iterator	\
			insert(typename delegated_container::nth_index<IndexIdT>::type::iterator hint,	\
				::Poseidon::Move<value_type> val)	\
		{	\
			return getIndex<IndexIdT>().insert(hint, STD_MOVE(val));	\
		}	\
		\
		template<unsigned IndexIdT>  \
		typename delegated_container::nth_index<IndexIdT>::type::iterator	\
			erase(typename delegated_container::nth_index<IndexIdT>::type::iterator pos)	\
		{	\
			return getIndex<IndexIdT>().erase(pos);	\
		}	\
		template<unsigned IndexIdT>  \
		typename delegated_container::nth_index<IndexIdT>::type::iterator	\
			erase(typename delegated_container::nth_index<IndexIdT>::type::iterator begin,	\
				typename delegated_container::nth_index<IndexIdT>::type::iterator end)	\
		{	\
			return getIndex<IndexIdT>().erase(begin, end);	\
		}	\
		template<unsigned IndexIdT>  \
		std::size_t erase(const typename delegated_container::nth_index<IndexIdT>::type::key_type &key){	\
			return getIndex<IndexIdT>().erase(key);	\
		}	\
		\
		template<unsigned IndexIdT> \
		bool replace(typename delegated_container::nth_index<IndexIdT>::type::iterator pos,	\
			const value_type &val)	\
		{	\
			return getIndex<IndexIdT>().replace(pos, val);	\
		}	\
		template<unsigned IndexIdT> \
		bool replace(typename delegated_container::nth_index<IndexIdT>::type::iterator pos,	\
			::Poseidon::Move<value_type> val)	\
		{	\
			return getIndex<IndexIdT>().replace(pos, STD_MOVE(val));	\
		}	\
		\
		template<unsigned IndexIdToSetT> \
		bool setKey(typename delegated_container::iterator pos,   \
			typename delegated_container::nth_index<IndexIdToSetT>::type::key_type key)	\
		{	\
			typename delegated_container::nth_index<IndexIdToSetT>::type::key_type old =	\
				typename delegated_container::nth_index<IndexIdToSetT>::type::key_from_value()(*pos);	\
			return getIndex<IndexIdToSetT>().modify_key(m_delegate.project<IndexIdToSetT>(pos),	\
				KeyModifier<IndexIdToSetT>(key), KeyModifier<IndexIdToSetT>(old));	\
		}	\
		template<unsigned IndexIdT, unsigned IndexIdToSetT> \
		bool setKey(typename delegated_container::nth_index<IndexIdT>::type::iterator pos,   \
			typename delegated_container::nth_index<IndexIdToSetT>::type::key_type key)	\
		{	\
			typename delegated_container::nth_index<IndexIdToSetT>::type::key_type old =	\
				typename delegated_container::nth_index<IndexIdToSetT>::type::key_from_value()(*pos);	\
			return getIndex<IndexIdToSetT>().modify_key(m_delegate.project<IndexIdToSetT>(pos),	\
				KeyModifier<IndexIdToSetT>(key), KeyModifier<IndexIdToSetT>(old));	\
		}	\
		\
		template<unsigned IndexIdT>  \
		typename delegated_container::nth_index<IndexIdT>::type::const_iterator	\
			find(const typename delegated_container::nth_index<IndexIdT>::type::key_type &key) const	\
		{	\
			return getIndex<IndexIdT>().find(key);	\
		}	\
		template<unsigned IndexIdT>  \
		typename delegated_container::nth_index<IndexIdT>::type::iterator	\
			find(const typename delegated_container::nth_index<IndexIdT>::type::key_type &key)	\
		{	\
			return getIndex<IndexIdT>().find(key);	\
		}	\
		\
		template<unsigned IndexIdT>  \
		::std::size_t	\
			count(const typename delegated_container::nth_index<IndexIdT>::type::key_type &key) const	\
		{	\
			return getIndex<IndexIdT>().count(key);	\
		}	\
		\
		template<unsigned IndexIdT>  \
		typename delegated_container::nth_index<IndexIdT>::type::const_iterator	\
			lowerBound(const typename delegated_container::nth_index<IndexIdT>::type::key_type &key) const	\
		{	\
			return getIndex<IndexIdT>().lower_bound(key);	\
		}	\
		template<unsigned IndexIdT>  \
		typename delegated_container::nth_index<IndexIdT>::type::iterator	\
			lowerBound(const typename delegated_container::nth_index<IndexIdT>::type::key_type &key)	\
		{	\
			return getIndex<IndexIdT>().lower_bound(key);	\
		}	\
		\
		template<unsigned IndexIdT>  \
		typename delegated_container::nth_index<IndexIdT>::type::const_iterator	\
			upperBound(const typename delegated_container::nth_index<IndexIdT>::type::key_type &key) const	\
		{	\
			return getIndex<IndexIdT>().upper_bound(key);	\
		}	\
		template<unsigned IndexIdT>  \
		typename delegated_container::nth_index<IndexIdT>::type::iterator	\
			upperBound(const typename delegated_container::nth_index<IndexIdT>::type::key_type &key)	\
		{	\
			return getIndex<IndexIdT>().upper_bound(key);	\
		}	\
		\
		template<unsigned IndexIdT>  \
		::std::pair<typename delegated_container::nth_index<IndexIdT>::type::const_iterator,	\
			typename delegated_container::nth_index<IndexIdT>::type::const_iterator>	\
			equalRange(const typename delegated_container::nth_index<IndexIdT>::type::key_type &key) const	\
		{	\
			return getIndex<IndexIdT>().equal_range(key);	\
		}	\
		template<unsigned IndexIdT>  \
		::std::pair<typename delegated_container::nth_index<IndexIdT>::type::iterator,	\
			typename delegated_container::nth_index<IndexIdT>::type::iterator>	\
			equalRange(const typename delegated_container::nth_index<IndexIdT>::type::key_type &key)	\
		{	\
			return getIndex<IndexIdT>().equal_range(key);	\
		}	\
	};	\
	\
	inline void swap(class_name_ &lhs, class_name_ &rhs) throw() {	\
		lhs.swap(rhs);	\
	}

#define UNIQUE_INDEX_(ignored_, ...)	\
	, ::boost::multi_index::ordered_unique<	\
		::boost::multi_index::identity<value_type>,	\
		## __VA_ARGS__>

#define MULTI_INDEX_(ignored_, ...)	\
	, ::boost::multi_index::ordered_non_unique<	\
		::boost::multi_index::identity<value_type>,	\
		## __VA_ARGS__>

#define UNIQUE_INDEX(...)	\
	UNIQUE_INDEX_(void, ## __VA_ARGS__)

#define MULTI_INDEX(...)	\
	MULTI_INDEX_(void, ## __VA_ARGS__)

#define UNIQUE_MEMBER_INDEX(member_, ...)	\
	, ::boost::multi_index::ordered_unique<	\
		::boost::multi_index::member<value_type,	\
			CV_VALUE_TYPE(DECLREF(value_type).member_),	\
			&value_type::member_	\
		>,	\
		## __VA_ARGS__>

#define MULTI_MEMBER_INDEX(member_, ...)	\
	, ::boost::multi_index::ordered_non_unique<	\
		::boost::multi_index::member<value_type,	\
			CV_VALUE_TYPE(DECLREF(value_type).member_),	\
			&value_type::member_	\
		>,	\
		## __VA_ARGS__>

#define SEQUENCED_INDEX()	\
	, ::boost::multi_index::sequenced<>

#endif
