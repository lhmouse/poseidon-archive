// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_MYSQL_CONNECTION_IMPL_
#	error Please do not #include "connection_impl.hpp".
#endif

#ifndef POSEIDON_MYSQL_CONNECTION_IMPL_HPP_
#define POSEIDON_MYSQL_CONNECTION_IMPL_HPP_

#include "../cxx_ver.hpp"
#include <vector>
#include <utility>
#include <boost/noncopyable.hpp>
#include <mysql/mysql.h>
#include "../raii.hpp"

namespace Poseidon {

struct MySqlCloser {
	CONSTEXPR ::MYSQL *operator()() const NOEXCEPT {
		return NULLPTR;
	}
	void operator()(::MYSQL *mySql) const NOEXCEPT {
		::mysql_close(mySql);
	}
};

struct MySqlResultDeleter {
	CONSTEXPR ::MYSQL_RES *operator()() const NOEXCEPT {
		return NULLPTR;
	}
	void operator()(::MYSQL_RES *result) const NOEXCEPT {
		::mysql_free_result(result);
	}
};

class MySqlDriver;
class MySqlThreadContext;

typedef std::vector<std::pair<const char *, std::size_t> > MySqlColumns;

class MySqlConnectionImpl : public MySqlConnection {
private:
	::MYSQL m_mySqlObject;
	ScopedHandle<MySqlCloser> m_mySql;

	ScopedHandle<MySqlResultDeleter> m_result;
	MySqlColumns m_columns;

	::MYSQL_ROW m_row;
	unsigned long *m_lengths;

public:
	MySqlConnectionImpl(const std::string &serverAddr, unsigned serverPort,
		const std::string &userName, const std::string &password, const std::string &schema,
		bool useSsl, const std::string &charset);
	~MySqlConnectionImpl();

public:
	void executeSql(const std::string &sql);
	void waitForResult();

	bool fetchRow();
	boost::int64_t getSigned(const char *column) const;
	boost::uint64_t getUnsigned(const char *column) const;
	double getDouble(const char *column) const;
	std::string getString(const char *column) const;
};

}

#endif
