#include "../../precompiled.hpp"
#include "object_base.hpp"
#include "../atomic.hpp"
#include "../utilities.hpp"
#include "../singletons/mysql_daemon.hpp"
#include "field.hpp"
using namespace Poseidon;

MySqlObjectBase::MySqlObjectBase(const char *table)
	: m_table(table), m_timeStamp(0)
{
}
