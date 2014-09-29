#ifndef POSEIDON_MODULE_HPP_
#define POSEIDON_MODULE_HPP_

#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>
#include "virtual_shared_from_this.hpp"

namespace Poseidon {

class Module : boost::noncopyable
	, public virtual VirtualSharedFromThis
{
public:
	static boost::shared_ptr<Module> load(const char *path);

protected:
	Module(){
	}

public:
	virtual ~Module() = 0;
};

}

#endif
