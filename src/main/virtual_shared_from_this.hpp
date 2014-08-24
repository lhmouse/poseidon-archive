#ifndef POSEIDON_VIRTUAL_SHARED_FROM_THIS_HPP_
#define POSEIDON_VIRTUAL_SHARED_FROM_THIS_HPP_

#include <cassert>
#include <boost/enable_shared_from_this.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_virtual_base_of.hpp>

namespace Poseidon {

class VirtualSharedFromThis
	: public boost::enable_shared_from_this<VirtualSharedFromThis>
{
public:
	// 定义在别处。参考源文件中的注释。
	virtual ~VirtualSharedFromThis() = 0;

public:
	template<typename Derived>
	typename boost::enable_if_c<
		boost::is_virtual_base_of<VirtualSharedFromThis, Derived>::value,
		boost::shared_ptr<const Derived>
	>::type
		virtualSharedFromThis() const
	{
		return boost::dynamic_pointer_cast<const Derived>(shared_from_this());
	}
	template<typename Derived>
	typename boost::enable_if_c<
		boost::is_virtual_base_of<VirtualSharedFromThis, Derived>::value,
		boost::shared_ptr<Derived>
	>::type
		virtualSharedFromThis()
	{
		return boost::dynamic_pointer_cast<Derived>(shared_from_this());
	}
};

}

#endif
