#ifndef POSEIDON_DATA_OBJECT_HPP_
#define POSEIDON_DATA_OBJECT_HPP_

#include "virtual_shared_from_this.hpp"
#include <map>
#include <string>

namespace Poseidon {

template<typename ValueT>
class DataField
	: public virtual VirtualSharedFromThis
{
};

class DataObject
	: public virtual VirtualSharedFromThis
{
private:
	
};

}

#endif
