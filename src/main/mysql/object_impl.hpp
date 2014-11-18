// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_MYSQL_OBJECT_IMPL_
#   error Please do not #include "mysql/object_impl.hpp".
#endif

#ifndef POSEIDON_MYSQL_OBJECT_IMPL_HPP_
#define POSEIDON_MYSQL_OBJECT_IMPL_HPP_

namespace Poseidon {

struct MySqlObjectImpl {
	static void *getContext(const MySqlObjectBase &obj){
		return obj.m_context;
	}
	static void setContext(const MySqlObjectBase &obj, void *context){
		obj.m_context = context;
	}
};

}

#endif
