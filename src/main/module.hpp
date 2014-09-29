#ifndef POSEIDON_MODULE_HPP_
#define POSEIDON_MODULE_HPP_

#include <string>
#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>
#include "virtual_shared_from_this.hpp"

namespace Poseidon {

class Module : boost::noncopyable
	, public virtual VirtualSharedFromThis
{
public:
	static boost::shared_ptr<Module> load(std::string path);

private:
	const std::string m_path;

protected:
	explicit Module(std::string path);

public:
	virtual ~Module() = 0;

public:
	const std::string &getPath() const {
		return m_path;
	}
};

}

#endif
