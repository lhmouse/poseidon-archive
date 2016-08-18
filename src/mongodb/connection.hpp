// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_MONGODB_CONNECTION_HPP_
#define POSEIDON_MONGODB_CONNECTION_HPP_

#include "../cxx_util.hpp"
#include <string>
#include <cstring>
#include <boost/cstdint.hpp>
#include <boost/shared_ptr.hpp>

namespace Poseidon {

class Uuid;

namespace MongoDb {
	class Oid;
	class BsonBuilder;

	class Connection: NONCOPYABLE {
	public:
		static boost::shared_ptr<Connection> create(const char *server_addr, unsigned server_port,
			bool slave_ok, const char *database);

		static boost::shared_ptr<Connection> create(const std::string &server_addr, unsigned server_port,
			bool slave_ok, const std::string &database)
		{
			return create(server_addr.c_str(), server_port, slave_ok, database.c_str());
		}

	public:
		virtual ~Connection() = 0;

	public:
		void execute_insert(const char *collection, const BsonBuilder &data);
		void execute_update(const char *collection, const BsonBuilder &query, bool upsert, bool update_all, const BsonBuilder &data);
		void execute_delete(const char *collection, const BsonBuilder &query, bool delete_all);
		void execute_query(const char *collection, const BsonBuilder &query, std::size_t begin = 0, std::size_t limit = 0x7FFFFFFF);
		void discard_result() NOEXCEPT;

		bool fetch_next();

		Oid get_oid() const;
		bool get_boolean(const char *name) const;
		boost::int64_t get_signed(const char *name) const;
		boost::uint64_t get_unsigned(const char *name) const;
		double get_double(const char *name) const;
		std::string get_string(const char *name) const;
		boost::uint64_t get_datetime(const char *name) const;
		Uuid get_uuid(const char *name) const;
		std::string get_blob(const char *name) const;
	};
}

}

#endif
