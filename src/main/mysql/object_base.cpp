// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "object_base.hpp"
#include "../singletons/mysql_daemon.hpp"
using namespace Poseidon;

MySqlObjectBase::EscapedString::EscapedString(const std::string &plain){
	m_escaped.reserve(plain.size() + 16);
	for(std::string::const_iterator it = plain.begin(); it != plain.end(); ++it){
		switch(*it){
		case 0:		m_escaped.append("\\0"); break;
		case '\r':	m_escaped.append("\\r"); break;
		case '\n':	m_escaped.append("\\n"); break;
		case 0x1A:	m_escaped.append("\\Z"); break;
		case '\'':	m_escaped.append("\\'"); break;
		case '\"':	m_escaped.append("\\\""); break;
		case '\\':	m_escaped.append("\\\\"); break;
		default:	m_escaped.push_back(*it); break;
		}
	}
}

MySqlObjectBase::MySqlObjectBase()
	: m_autoSaves(false), m_context()
{
}
MySqlObjectBase::~MySqlObjectBase(){
}

void MySqlObjectBase::invalidate() const {
	if(!isAutoSavingEnabled()){
		return;
	}
	MySqlDaemon::pendForSaving(virtualSharedFromThis<MySqlObjectBase>());
}

void MySqlObjectBase::asyncSave() const {
	enableAutoSaving();
	MySqlDaemon::pendForSaving(virtualSharedFromThis<MySqlObjectBase>());
}
void MySqlObjectBase::asyncLoad(std::string filter, MySqlAsyncLoadCallback callback){
	disableAutoSaving();
	MySqlDaemon::pendForLoading(
		virtualSharedFromThis<MySqlObjectBase>(), STD_MOVE(filter), STD_MOVE(callback));
}
