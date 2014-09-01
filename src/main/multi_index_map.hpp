#ifndef POSEIDON_MULTI_INDEX_MAP_HPP_
#define POSEIDON_MULTI_INDEX_MAP_HPP_

/*

定义时需要指定索引。

typedef std::pair<int, std::string> Item;

MULTI_INDEX_MAP(Container, Item,
	UNIQUE_INDEX(first),
	MULTI_INDEX(second)
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

#define MULTI_INDEX_MAP(class_name_, value_type_, ...)	\
	class class_name_ {	\
	public:	\
		typedef value_type_ value_type;	\
		\
		typedef ::boost::multi_index::multi_index_container<	\
			value_type,	\
			::boost::multi_index::indexed_by<__VA_ARGS__>	\
		> delegate_container;	\
		\
	private:	\
		template<unsigned long IndexId>	\
		struct KeyModifier {	\
			typedef	\
				typename delegate_container::nth_index<IndexId>::type::key_type	\
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
				std::swap(old, m_key);	\
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
		template<unsigned long IndexId>  \
		const typename delegate_container::nth_index<IndexId>::type &	\
			getIndex() const	\
		{	\
			return m_delegate.get<IndexId>();	\
		}	\
		template<unsigned long IndexId>  \
		typename delegate_container::nth_index<IndexId>::type &	\
			getIndex()	\
		{	\
			return m_delegate.get<IndexId>();	\
		}	\
		\
		std::pair<delegate_container::iterator, bool>	\
			insert(const value_type &val)	\
		{	\
			return m_delegate.insert(val);	\
		}	\
		\
		template<unsigned long IndexId>	\
		std::pair<typename delegate_container::nth_index<IndexId>::type::iterator, bool>	\
			insert(typename delegate_container::nth_index<IndexId>::type::iterator hint,	\
				const value_type &val)	\
		{	\
			return getIndex<IndexId>().insert(hint, val);	\
		}	\
		\
		template<unsigned IndexId>  \
		void erase(typename delegate_container::nth_index<IndexId>::type::iterator pos){	\
			getIndex<IndexId>().erase(pos);	\
		}	\
		\
		template<unsigned long IndexId> \
		bool replace(typename delegate_container::nth_index<IndexId>::type::iterator pos,	\
			const value_type &val)  \
		{	\
			return getIndex<IndexId>().replace(pos, val);	\
		}	\
		template<unsigned long IndexId, unsigned long IndexToSet> \
		bool setKey(typename delegate_container::nth_index<IndexId>::type::iterator pos,   \
			typename delegate_container::nth_index<IndexToSet>::type::key_type key)	\
		{	\
			typename delegate_container::nth_index<IndexToSet>::type::key_type old =	\
				typename delegate_container::nth_index<IndexToSet>::type::key_from_value()(*pos);	\
			return getIndex<IndexToSet>().modify_key(	\
				m_delegate.project<IndexToSet>(pos),	\
				KeyModifier<IndexToSet>(key),	\
				KeyModifier<IndexToSet>(old)	\
			);	\
		}	\
		\
		template<unsigned IndexId>  \
		typename delegate_container::nth_index<IndexId>::type::const_iterator   \
			begin() const	\
		{	\
			return getIndex<IndexId>().begin();	\
		}	\
		template<unsigned IndexId>  \
		typename delegate_container::nth_index<IndexId>::type::iterator   \
			begin()	\
		{	\
			return getIndex<IndexId>().begin();	\
		}	\
		\
		template<unsigned IndexId>  \
		typename delegate_container::nth_index<IndexId>::type::const_iterator   \
			end() const	\
		{	\
			return getIndex<IndexId>().end();	\
		}	\
		template<unsigned IndexId>  \
		typename delegate_container::nth_index<IndexId>::type::iterator   \
			end()	\
		{	\
			return getIndex<IndexId>().end();	\
		}	\
		\
		template<unsigned IndexId>  \
		typename delegate_container::nth_index<IndexId>::type::const_iterator	\
			find(const typename delegate_container::nth_index<IndexId>::type::key_type &key) const	\
		{	\
			return getIndex<IndexId>().find(key);	\
		}	\
		template<unsigned IndexId>  \
		typename delegate_container::nth_index<IndexId>::type::iterator	\
			find(const typename delegate_container::nth_index<IndexId>::type::key_type &key)	\
		{	\
			return getIndex<IndexId>().find(key);	\
		}	\
		\
		template<unsigned IndexId>  \
		std::size_t	\
			count(const typename delegate_container::nth_index<IndexId>::type::key_type &key) const	\
		{	\
			return getIndex<IndexId>().count(key);	\
		}	\
		\
		template<unsigned IndexId>  \
		typename delegate_container::nth_index<IndexId>::type::const_iterator	\
			lowerBound(const typename delegate_container::nth_index<IndexId>::type::key_type &key) const	\
		{	\
			return getIndex<IndexId>().lower_bound(key);	\
		}	\
		template<unsigned IndexId>  \
		typename delegate_container::nth_index<IndexId>::type::iterator	\
			lowerBound(const typename delegate_container::nth_index<IndexId>::type::key_type &key)	\
		{	\
			return getIndex<IndexId>().lower_bound(key);	\
		}	\
		\
		template<unsigned IndexId>  \
		typename delegate_container::nth_index<IndexId>::type::const_iterator	\
			upperBound(const typename delegate_container::nth_index<IndexId>::type::key_type &key) const	\
		{	\
			return getIndex<IndexId>().upper_bound(key);	\
		}	\
		template<unsigned IndexId>  \
		typename delegate_container::nth_index<IndexId>::type::iterator	\
			upperBound(const typename delegate_container::nth_index<IndexId>::type::key_type &key)	\
		{	\
			return getIndex<IndexId>().upper_bound(key);	\
		}	\
		\
		template<unsigned IndexId>  \
		std::pair<typename delegate_container::nth_index<IndexId>::type::const_iterator,	\
			typename delegate_container::nth_index<IndexId>::type::const_iterator	\
		>	\
			equalRange(const typename delegate_container::nth_index<IndexId>::type::key_type &key) const	\
		{	\
			return getIndex<IndexId>().equal_range(key);	\
		}	\
		template<unsigned IndexId>  \
		std::pair<typename delegate_container::nth_index<IndexId>::type::iterator,	\
			typename delegate_container::nth_index<IndexId>::type::iterator	\
		>	\
			equalRange(const typename delegate_container::nth_index<IndexId>::type::key_type &key)	\
		{	\
			return getIndex<IndexId>().equal_range(key);	\
		}	\
	};

#define MULTI_INDEX_MAP_DECLTYPE_(struct_, member_)	\
	__typeof__(((struct_ *)1)->member_)

#define UNIQUE_INDEX(member_)	\
	::boost::multi_index::ordered_unique<	\
		::boost::multi_index::member<value_type,	\
			MULTI_INDEX_MAP_DECLTYPE_(value_type, member_),	\
			&value_type::member_	\
		>	\
	>

#define MULTI_INDEX(member_)	\
	::boost::multi_index::ordered_non_unique<	\
		::boost::multi_index::member<value_type,	\
			MULTI_INDEX_MAP_DECLTYPE_(value_type, member_),	\
			&value_type::member_	\
		>	\
	>

#endif
