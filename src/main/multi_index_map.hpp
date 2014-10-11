#ifndef POSEIDON_MULTI_INDEX_MAP_HPP_
#define POSEIDON_MULTI_INDEX_MAP_HPP_

/*

定义时需要指定索引。

typedef ::std::pair<int, ::std::string> Item;

MULTI_INDEX_MAP(Container, Item,
	UNIQUE_MEMBER_INDEX(first),
	MULTI_MEMBER_INDEX(second),
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

#include "../cxx_ver.hpp"
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/identity.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/sequenced_index.hpp>

#define MULTI_INDEX_MAP(class_name_, value_type_, ...)	\
	class class_name_ {	\
	public:	\
		typedef value_type_ value_type;	\
		\
		typedef ::boost::multi_index::multi_index_container<	\
			value_type, ::boost::multi_index::indexed_by<__VA_ARGS__>	\
			> delegate_container;	\
		\
		typedef delegate_container::const_iterator const_iterator;	\
		typedef delegate_container::iterator iterator;	\
		typedef delegate_container::const_reverse_iterator const_reverse_iterator;	\
		typedef delegate_container::reverse_iterator reverse_iterator;	\
	private:	\
		template<unsigned IndexIdT>	\
		struct KeyModifier {	\
			typedef	\
				typename delegate_container::nth_index<IndexIdT>::type::key_type	\
				key_type;	\
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
		delegate_container m_delegate;	\
		\
	public:	\
		template<unsigned IndexIdT>  \
		const typename delegate_container::nth_index<IndexIdT>::type &getIndex() const {	\
			return m_delegate.get<IndexIdT>();	\
		}	\
		template<unsigned IndexIdT>  \
		typename delegate_container::nth_index<IndexIdT>::type &getIndex(){	\
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
		typename delegate_container::nth_index<IndexIdT>::type::const_iterator	\
			begin() const	\
		{	\
			return getIndex<IndexIdT>().begin();	\
		}	\
		template<unsigned IndexIdT>  \
		typename delegate_container::nth_index<IndexIdT>::type::iterator	\
			begin()	\
		{	\
			return getIndex<IndexIdT>().begin();	\
		}	\
		template<unsigned IndexIdT>  \
		typename delegate_container::nth_index<IndexIdT>::type::const_iterator	\
			end() const	\
		{	\
			return getIndex<IndexIdT>().end();	\
		}	\
		template<unsigned IndexIdT>  \
		typename delegate_container::nth_index<IndexIdT>::type::iterator	\
			end()	\
		{	\
			return getIndex<IndexIdT>().end();	\
		}	\
		\
		template<unsigned IndexIdT>  \
		typename delegate_container::nth_index<IndexIdT>::type::const_reverse_iterator	\
			rbegin() const	\
		{	\
			return getIndex<IndexIdT>().rbegin();	\
		}	\
		template<unsigned IndexIdT>  \
		typename delegate_container::nth_index<IndexIdT>::type::reverse_iterator	\
			rbegin()	\
		{	\
			return getIndex<IndexIdT>().rbegin();	\
		}	\
		template<unsigned IndexIdT>  \
		typename delegate_container::nth_index<IndexIdT>::type::const_reverse_iterator	\
			rend() const	\
		{	\
			return getIndex<IndexIdT>().rend();	\
		}	\
		template<unsigned IndexIdT>  \
		typename delegate_container::nth_index<IndexIdT>::type::reverse_iterator	\
			rend()	\
		{	\
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
		iterator insert(iterator pos, const value_type &val){	\
			return m_delegate.insert(pos, val);	\
		}	\
	ENABLE_IF_CXX11(	\
		::std::pair<iterator, bool> insert(value_type &&val){	\
			return m_delegate.insert(::std::move(val));	\
		}	\
		iterator insert(iterator pos, value_type &&val){	\
			return m_delegate.insert(pos, ::std::move(val));	\
		}	\
	)	\
		iterator erase(iterator pos){	\
			iterator ret = pos;	\
			++ret;	\
			m_delegate.erase(pos);	\
			return ret;	\
		}	\
		\
		template<unsigned IndexIdT>	\
		::std::pair<typename delegate_container::nth_index<IndexIdT>::type::iterator, bool>	\
			insert(const value_type &val)	\
		{	\
			return getIndex<IndexIdT>().insert(val);	\
		}	\
		template<unsigned IndexIdT>	\
		typename delegate_container::nth_index<IndexIdT>::type::iterator	\
			insert(	\
				typename delegate_container::nth_index<IndexIdT>::type::iterator hint,	\
				const value_type &val	\
			)	\
		{	\
			return getIndex<IndexIdT>().insert(hint, val);	\
		}	\
	ENABLE_IF_CXX11(	\
		template<unsigned IndexIdT>	\
		::std::pair<typename delegate_container::nth_index<IndexIdT>::type::iterator, bool>	\
			insert(value_type &&val)	\
		{	\
			return getIndex<IndexIdT>().insert(::std::move(val));	\
		}	\
		template<unsigned IndexIdT>	\
		typename delegate_container::nth_index<IndexIdT>::type::iterator	\
			insert(	\
				typename delegate_container::nth_index<IndexIdT>::type::iterator hint,	\
				value_type &&val	\
			)	\
		{	\
			return getIndex<IndexIdT>().insert(hint, ::std::move(val));	\
		}	\
	)	\
		template<unsigned IndexIdT>  \
		typename delegate_container::nth_index<IndexIdT>::type::iterator	\
			erase(typename delegate_container::nth_index<IndexIdT>::type::iterator pos)	\
		{	\
			typename delegate_container::nth_index<IndexIdT>::type::iterator ret = pos;	\
			++ret;	\
			getIndex<IndexIdT>().erase(pos);	\
			return ret;	\
		}	\
		template<unsigned IndexIdT>  \
		void erase(const typename delegate_container::nth_index<IndexIdT>::type::key_type &key){	\
			getIndex<IndexIdT>().erase(key);	\
		}	\
		\
		template<unsigned IndexIdT> \
		bool replace(	\
			typename delegate_container::nth_index<IndexIdT>::type::iterator pos,	\
			const value_type &val	\
		){	\
			return getIndex<IndexIdT>().replace(pos, val);	\
		}	\
	ENABLE_IF_CXX11(	\
		template<unsigned IndexIdT> \
		bool replace(	\
			typename delegate_container::nth_index<IndexIdT>::type::iterator pos,	\
			value_type &&val	\
		){	\
			return getIndex<IndexIdT>().replace(pos, ::std::move(val));	\
		}	\
	)	\
		template<unsigned IndexIdToSetT> \
		bool setKey(	\
			typename delegate_container::iterator pos,   \
			typename delegate_container::nth_index<IndexIdToSetT>::type::key_type key	\
		){	\
			typename delegate_container::nth_index<IndexIdToSetT>::type::key_type old =	\
				typename delegate_container::nth_index<IndexIdToSetT>::type::key_from_value()(*pos);	\
			return getIndex<IndexIdToSetT>().modify_key(m_delegate.project<IndexIdToSetT>(pos),	\
				KeyModifier<IndexIdToSetT>(key), KeyModifier<IndexIdToSetT>(old)	\
			);	\
		}	\
		template<unsigned IndexIdT, unsigned IndexIdToSetT> \
		bool setKey(	\
			typename delegate_container::nth_index<IndexIdT>::type::iterator pos,   \
			typename delegate_container::nth_index<IndexIdToSetT>::type::key_type key	\
		){	\
			typename delegate_container::nth_index<IndexIdToSetT>::type::key_type old =	\
				typename delegate_container::nth_index<IndexIdToSetT>::type::key_from_value()(*pos);	\
			return getIndex<IndexIdToSetT>().modify_key(m_delegate.project<IndexIdToSetT>(pos),	\
				KeyModifier<IndexIdToSetT>(key), KeyModifier<IndexIdToSetT>(old)	\
			);	\
		}	\
		\
		template<unsigned IndexIdT>  \
		typename delegate_container::nth_index<IndexIdT>::type::const_iterator	\
			find(	\
				const typename delegate_container::nth_index<IndexIdT>::type::key_type &key	\
			) const	\
		{	\
			return getIndex<IndexIdT>().find(key);	\
		}	\
		template<unsigned IndexIdT>  \
		typename delegate_container::nth_index<IndexIdT>::type::iterator	\
			find(	\
				const typename delegate_container::nth_index<IndexIdT>::type::key_type &key	\
			)	\
		{	\
			return getIndex<IndexIdT>().find(key);	\
		}	\
		\
		template<unsigned IndexIdT>  \
		::std::size_t	\
			count(	\
				const typename delegate_container::nth_index<IndexIdT>::type::key_type &key	\
			) const	\
		{	\
			return getIndex<IndexIdT>().count(key);	\
		}	\
		\
		template<unsigned IndexIdT>  \
		typename delegate_container::nth_index<IndexIdT>::type::const_iterator	\
			lowerBound(	\
				const typename delegate_container::nth_index<IndexIdT>::type::key_type &key	\
			) const	\
		{	\
			return getIndex<IndexIdT>().lower_bound(key);	\
		}	\
		template<unsigned IndexIdT>  \
		typename delegate_container::nth_index<IndexIdT>::type::iterator	\
			lowerBound(	\
				const typename delegate_container::nth_index<IndexIdT>::type::key_type &key	\
			)	\
		{	\
			return getIndex<IndexIdT>().lower_bound(key);	\
		}	\
		\
		template<unsigned IndexIdT>  \
		typename delegate_container::nth_index<IndexIdT>::type::const_iterator	\
			upperBound(	\
				const typename delegate_container::nth_index<IndexIdT>::type::key_type &key	\
			) const	\
		{	\
			return getIndex<IndexIdT>().upper_bound(key);	\
		}	\
		template<unsigned IndexIdT>  \
		typename delegate_container::nth_index<IndexIdT>::type::iterator	\
			upperBound(	\
				const typename delegate_container::nth_index<IndexIdT>::type::key_type &key	\
			)	\
		{	\
			return getIndex<IndexIdT>().upper_bound(key);	\
		}	\
		\
		template<unsigned IndexIdT>  \
		::std::pair<typename delegate_container::nth_index<IndexIdT>::type::const_iterator,	\
			typename delegate_container::nth_index<IndexIdT>::type::const_iterator>	\
			equalRange(	\
				const typename delegate_container::nth_index<IndexIdT>::type::key_type &key	\
			) const	\
		{	\
			return getIndex<IndexIdT>().equal_range(key);	\
		}	\
		template<unsigned IndexIdT>  \
		::std::pair<typename delegate_container::nth_index<IndexIdT>::type::iterator,	\
			typename delegate_container::nth_index<IndexIdT>::type::iterator>	\
			equalRange(	\
				const typename delegate_container::nth_index<IndexIdT>::type::key_type &key	\
			)	\
		{	\
			return getIndex<IndexIdT>().equal_range(key);	\
		}	\
	};	\
	\
	static inline void swap(class_name_ &lhs, class_name_ &rhs) throw() {	\
		lhs.swap(rhs);	\
	}

#define MULTI_INDEX_MAP_DECLTYPE_(struct_, member_)	\
	__typeof__(((struct_ *)1)->member_)

#define UNIQUE_INDEX_(ignored_, ...)	\
	::boost::multi_index::ordered_unique<	\
		::boost::multi_index::identity<value_type>,	\
		## __VA_ARGS__	\
	>

#define MULTI_INDEX_(ignored_, ...)	\
	::boost::multi_index::ordered_non_unique<	\
		::boost::multi_index::identity<value_type>,	\
		## __VA_ARGS__	\
	>

#define UNIQUE_INDEX(...)	\
	UNIQUE_INDEX_(void, ## __VA_ARGS__)

#define MULTI_INDEX(...)	\
	MULTI_INDEX_(void, ## __VA_ARGS__)

#define UNIQUE_MEMBER_INDEX(member_, ...)	\
	::boost::multi_index::ordered_unique<	\
		::boost::multi_index::member<value_type,	\
			MULTI_INDEX_MAP_DECLTYPE_(value_type, member_),	\
			&value_type::member_	\
		>,	\
		## __VA_ARGS__	\
	>

#define MULTI_MEMBER_INDEX(member_, ...)	\
	::boost::multi_index::ordered_non_unique<	\
		::boost::multi_index::member<value_type,	\
			MULTI_INDEX_MAP_DECLTYPE_(value_type, member_),	\
			&value_type::member_	\
		>,	\
		## __VA_ARGS__	\
	>

#define SEQUENCED_INDEX()	\
	::boost::multi_index::sequenced<>

#endif
