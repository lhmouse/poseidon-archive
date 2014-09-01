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
	template<typename Derived>
	boost::shared_ptr<const Derived> virtualSharedFromThis() const {
		BOOST_STATIC_ASSERT_MSG(
			(boost::is_virtual_base_of<VirtualSharedFromThis, Derived>::value),
			"Please virtually derive from VirtualSharedFromThis.");

		return boost::dynamic_pointer_cast<const Derived>(shared_from_this());
	}
	template<typename Derived>
	boost::shared_ptr<Derived> virtualSharedFromThis(){
		BOOST_STATIC_ASSERT_MSG(
			(boost::is_virtual_base_of<VirtualSharedFromThis, Derived>::value),
			"Please virtually derive from VirtualSharedFromThis.");

		return boost::dynamic_pointer_cast<Derived>(shared_from_this());
	}

	template<typename Derived>
	boost::weak_ptr<const Derived> virtualWeakFromThis() const {
		return virtualSharedFromThis<const Derived>();
	}
	template<typename Derived>
	boost::weak_ptr<Derived> virtualWeakFromThis(){
		return virtualSharedFromThis<Derived>();
	}
};

}

#endif
