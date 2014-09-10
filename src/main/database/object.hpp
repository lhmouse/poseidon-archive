#ifndef POSEIDON_DATABASE_OBJECT_HPP_
#define POSEIDON_DATABASE_OBJECT_HPP_

#include "../../cxx_ver.hpp"
#include "../virtual_shared_from_this.hpp"
#include <map>
#include <string>

namespace Poseidon {

class DatabaseObject;

template<typename ValueT>
class DatabaseField
	: public virtual VirtualSharedFromThis
{
private:
	ValueT m_val;

public:
	explicit DatabaseField(DatabaseObject *owner, ValueT val = ValueT())
		: m_val(STD_MOVE(val))
	{
	}

public:
	const ValueT &get() const {
		return m_val;
	}
	void set(ValueT val){
		using std::swap;
		swap(val, m_val);
	}
};

class DatabaseObject
	: public virtual VirtualSharedFromThis
{
private:
	
};

}

#endif
