#include "../../precompiled.hpp"
#include "object_base.hpp"
#include <cppconn/connection.h>
#include <cppconn/prepared_statement.h>
#include <cppconn/resultset.h>
#include "../log.hpp"
#include "../job_base.hpp"
#include "../singletons/mysql_daemon.hpp"
using namespace Poseidon;

class MySqlObjectBase::AsyncLoadJob : public JobBase {
};

class MySqlObjectBase::AsyncSaveJob : public JobBase {
};

void MySqlObjectBase::asyncLoad(std::string filter, MySqlObjectBase::AsyncCallback callback){
	(void)filter;
	(void)callback;
}
void MySqlObjectBase::asyncSave(AsyncCallback callback){
	(void)callback;
}
