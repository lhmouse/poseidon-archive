// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_VIRTUAL_SHARED_FROM_THIS_HPP_
#define POSEIDON_VIRTUAL_SHARED_FROM_THIS_HPP_

#include <cassert>
#include <boost/enable_shared_from_this.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/type_traits/is_virtual_base_of.hpp>
#include <boost/static_assert.hpp>

namespace Poseidon {

class VirtualSharedFromThis
	: public boost::enable_shared_from_this<VirtualSharedFromThis>
{
public:
	// 定义在别处。参考源文件中的注释。
	virtual ~VirtualSharedFromThis() = 0;

public:
	template<typename DerivedT>
	boost::shared_ptr<const DerivedT> virtual_shared_from_this() const {
		BOOST_STATIC_ASSERT((boost::is_virtual_base_of<VirtualSharedFromThis, DerivedT>::value));

		return boost::shared_ptr<const DerivedT>(shared_from_this(), &dynamic_cast<const DerivedT &>(*this));
	}
	template<typename DerivedT>
	boost::shared_ptr<DerivedT> virtual_shared_from_this(){
		BOOST_STATIC_ASSERT((boost::is_virtual_base_of<VirtualSharedFromThis, DerivedT>::value));

		return boost::shared_ptr<DerivedT>(shared_from_this(), &dynamic_cast<DerivedT &>(*this));
	}

	template<typename DerivedT>
	boost::weak_ptr<const DerivedT> virtual_weak_from_this() const {
		return virtual_shared_from_this<const DerivedT>();
	}
	template<typename DerivedT>
	boost::weak_ptr<DerivedT> virtual_weak_from_this(){
		return virtual_shared_from_this<DerivedT>();
	}
};

}

#endif
