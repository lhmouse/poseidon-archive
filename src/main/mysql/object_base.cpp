#include "../../precompiled.hpp"
#include "object_base.hpp"
#include <cppconn/connection.h>
#include <cppconn/prepared_statement.h>
#include <cppconn/resultset.h>
#include "../log.hpp"
#include "../singletons/mysql_daemon.hpp"
#include "field.hpp"
using namespace Poseidon;

MySqlObjectBase::MySqlObjectBase(const char *table)
	: m_table(table)
{
}

void MySqlObjectBase::syncLoad(sql::Connection *conn, const std::string &filter){
	if(m_fields.empty()){
		return;
	}

	std::string sql;
	sql = "SELECT ";
	for(AUTO(it, m_fields.begin()); it != m_fields.end(); ++it){
		sql += '`';
		sql += it->get().name();
		sql += "`, ";
	}
	sql.erase(sql.end() - 2, sql.end());
	sql += " FROM `";
	sql += m_table;
	sql += "` ";
	sql += filter;
	sql += " LIMIT 1";

	LOG_DEBUG("Sync load: SQL = ", sql);

	const boost::scoped_ptr<sql::PreparedStatement> stmt(conn->prepareStatement(sql));
	const boost::scoped_ptr<sql::ResultSet> rs(stmt->executeQuery());
	if(rs->first()){
		for(std::size_t i = 0; i < m_fields.size(); ++i){
			m_fields[i].get().fetch(i, rs.get());
		}
	} else {
		LOG_DEBUG("Empty set returned.");
	}
}
void MySqlObjectBase::syncSave(sql::Connection *conn){
	std::string sql;
	sql = "REPLACE INTO `";
	sql += m_table;
	sql += "`(";
	for(AUTO(it, m_fields.begin()); it != m_fields.end(); ++it){
		sql += '`';
		sql += it->get().name();
		sql += "`, ";
	}
	sql.erase(sql.end() - 2, sql.end());
	sql += ") VALUES(";
	for(AUTO(it, m_fields.begin()); it != m_fields.end(); ++it){
		sql += "?, ";
	}
	sql.erase(sql.end() - 2, sql.end());
	sql += ')';

	LOG_DEBUG("Sync save: SQL = ", sql);

	const boost::scoped_ptr<sql::PreparedStatement> ps(conn->prepareStatement(sql));
	std::vector<boost::any> contexts;
	for(std::size_t i = 0; i < m_fields.size(); ++i){
		m_fields[i].get().pack(i, ps.get(), contexts);
	}
	ps->executeUpdate();
}
