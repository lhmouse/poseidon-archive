#ifndef POSEIDON_MYSQL_OBJECT_IMPL_
#   error Please do not #include "mysql/object_impl.hpp".
#endif

#ifndef POSEIDON_MYSQL_OBJECT_IMPL_HPP_
#define POSEIDON_MYSQL_OBJECT_IMPL_HPP_

namespace Poseidon {

struct MySqlObjectImpl {
	static boost::shared_ptr<Module> getModule(const MySqlObjectBase &obj){
		return obj.m_module;
	}
	static void *getContext(const MySqlObjectBase &obj){
		return obj.m_context;
	}
	static void setContext(const MySqlObjectBase &obj, void *context){
		obj.m_context = context;
	}
};

}

#endif
