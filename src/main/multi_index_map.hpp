#ifndef POSEIDON_MULTI_INDEX_MAP_HPP_
#define POSEIDON_MULTI_INDEX_MAP_HPP_

/*

定义时需要指定索引。

typedef std::pair<int, std::string> Item;

MULTI_INDEX_MAP(Container, Item,
	UNIQUE_INDEX(first),
	MULTI_INDEX(second),
	SEQUENCED_INDEX()
);

基本用法和 std::map 类似，只是 find, count, lowerBound, upperBound, equalRange 等
成员函数需要带一个非类型模板参数，指定使用第几个索引。

Container c;
c.insert(Item(1, "abc"));
c.insert(Item(2, "def"));
std::cout <<c.find<0>(1)->second <<std::endl;	// "abc";
assert(c.upperBound<1>("zzz") == c.end<1>());	// 通过。

*/

#include <utility>
#include <boost/multi_index_container.hpp>
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
				using std::swap;	\
				swap(old, m_key);	\
			}	\
		};	\
		\
	private:	\
		delegate_container m_delegate;	\
		\
	public:	\
		bool empty() const throw() {	\
			return m_delegate.empty();	\
		}	\
		std::size_t size() const throw() {	\
			return m_delegate.size();	\
		}	\
		void clear() throw() {	\
			m_delegate.clear();	\
		}	\
		\
		template<unsigned IndexIdT>  \
		const typename delegate_container::nth_index<IndexIdT>::type &getIndex() const {	\
			return m_delegate.get<IndexIdT>();	\
		}	\
		template<unsigned IndexIdT>  \
		typename delegate_container::nth_index<IndexIdT>::type &getIndex(){	\
			return m_delegate.get<IndexIdT>();	\
		}	\
		\
		std::pair<delegate_container::iterator, bool> insert(const value_type &val){	\
			return m_delegate.insert(val);	\
		}	\
		\
		template<unsigned IndexIdT>	\
		std::pair<typename delegate_container::nth_index<IndexIdT>::type::iterator, bool>	\
			insert(	\
				typename delegate_container::nth_index<IndexIdT>::type::iterator hint,	\
				const value_type &val	\
			)	\
		{	\
			return getIndex<IndexIdT>().insert(hint, val);	\
		}	\
		\
		template<unsigned IndexIdT>  \
		void erase(typename delegate_container::nth_index<IndexIdT>::type::iterator pos){	\
			getIndex<IndexIdT>().erase(pos);	\
		}	\
		\
		template<unsigned IndexIdT> \
		bool replace(	\
			typename delegate_container::nth_index<IndexIdT>::type::iterator pos,	\
			const value_type &val	\
		){	\
			return getIndex<IndexIdT>().replace(pos, val);	\
		}	\
		template<unsigned IndexIdT, unsigned IndexIdoSetT> \
		bool setKey(	\
			typename delegate_container::nth_index<IndexIdT>::type::iterator pos,   \
			typename delegate_container::nth_index<IndexIdoSetT>::type::key_type key	\
		){	\
			typename delegate_container::nth_index<IndexIdoSetT>::type::key_type old =	\
				typename delegate_container::nth_index<IndexIdoSetT>::type::key_from_value()(*pos);	\
			return getIndex<IndexIdoSetT>().modify_key(m_delegate.project<IndexIdoSetT>(pos),	\
				KeyModifier<IndexIdoSetT>(key), KeyModifier<IndexIdoSetT>(old)	\
			);	\
		}	\
		\
		template<unsigned IndexIdT>  \
		typename delegate_container::nth_index<IndexIdT>::type::const_iterator begin() const {	\
			return getIndex<IndexIdT>().begin();	\
		}	\
		template<unsigned IndexIdT>  \
		typename delegate_container::nth_index<IndexIdT>::type::iterator begin(){	\
			return getIndex<IndexIdT>().begin();	\
		}	\
		\
		template<unsigned IndexIdT>  \
		typename delegate_container::nth_index<IndexIdT>::type::const_iterator end() const {	\
			return getIndex<IndexIdT>().end();	\
		}	\
		template<unsigned IndexIdT>  \
		typename delegate_container::nth_index<IndexIdT>::type::iterator end(){	\
			return getIndex<IndexIdT>().end();	\
		}	\
		\
		template<unsigned IndexIdT>  \
		typename delegate_container::nth_index<IndexIdT>::type::const_iterator rbegin() const {	\
			return getIndex<IndexIdT>().rbegin();	\
		}	\
		template<unsigned IndexIdT>  \
		typename delegate_container::nth_index<IndexIdT>::type::iterator rbegin(){	\
			return getIndex<IndexIdT>().rbegin();	\
		}	\
		\
		template<unsigned IndexIdT>  \
		typename delegate_container::nth_index<IndexIdT>::type::const_iterator rend() const {	\
			return getIndex<IndexIdT>().rend();	\
		}	\
		template<unsigned IndexIdT>  \
		typename delegate_container::nth_index<IndexIdT>::type::iterator rend(){	\
			return getIndex<IndexIdT>().rend();	\
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
		std::size_t	\
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
		std::pair<typename delegate_container::nth_index<IndexIdT>::type::const_iterator,	\
			typename delegate_container::nth_index<IndexIdT>::type::const_iterator>	\
			equalRange(	\
				const typename delegate_container::nth_index<IndexIdT>::type::key_type &key	\
			) const	\
		{	\
			return getIndex<IndexIdT>().equal_range(key);	\
		}	\
		template<unsigned IndexIdT>  \
		std::pair<typename delegate_container::nth_index<IndexIdT>::type::iterator,	\
			typename delegate_container::nth_index<IndexIdT>::type::iterator>	\
			equalRange(	\
				const typename delegate_container::nth_index<IndexIdT>::type::key_type &key	\
			)	\
		{	\
			return getIndex<IndexIdT>().equal_range(key);	\
		}	\
	};

#define MULTI_INDEX_MAP_DECLTYPE_(struct_, member_)	\
	__typeof__(((struct_ *)1)->member_)

#define UNIQUE_INDEX(member_, ...)	\
	::boost::multi_index::ordered_unique<	\
		::boost::multi_index::member<value_type,	\
			MULTI_INDEX_MAP_DECLTYPE_(value_type, member_),	\
			&value_type::member_	\
		>,	\
		## __VA_ARGS__	\
	>

#define MULTI_INDEX(member_, ...)	\
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
