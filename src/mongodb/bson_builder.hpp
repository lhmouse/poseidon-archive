// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_MONGODB_BSON_BUILDER_HPP_
#define POSEIDON_MONGODB_BSON_BUILDER_HPP_

#include "../cxx_ver.hpp"
#include <deque>
#include <string>
#include <cstddef>
#include "../shared_nts.hpp"

namespace Poseidon {

class Uuid;

namespace MongoDb {
	class Oid;

	class BsonBuilder {
	private:
		enum Type {
			T_OID      =  0,
			T_BOOLEAN  =  1,
			T_SIGNED   =  2,
			T_UNSIGNED =  3,
			T_DOUBLE   =  4,
			T_STRING   =  5,
			T_DATETIME =  6,
			T_UUID     =  7,
			T_BLOB     =  8,
			T_REGEX    =  9,
			T_OBJECT   = 10,
			T_ARRAY    = 11,
		};

		struct Element {
			Type type;
			SharedNts name;
			char small[36];
			std::string large;
		};

	private:
		std::deque<Element> m_queue;

	public:
		void append_oid(const Oid &oid);
		void append_boolean(SharedNts name, bool value);
		void append_signed(SharedNts name, boost::int64_t value);
		void append_unsigned(SharedNts name, boost::uint64_t value);
		void append_double(SharedNts name, double value);
		void append_string(SharedNts name, std::string value);
		void append_datetime(SharedNts name, boost::uint64_t value);
		void append_uuid(SharedNts name, const Uuid &value);
		void append_blob(SharedNts name, std::string value);

		void append_regex(SharedNts name, std::string value, const char *options = "");
		void append_object(SharedNts name, const BsonBuilder &value);
		void append_array(SharedNts name, const BsonBuilder &value);

		void clear() NOEXCEPT {
			m_queue.clear();
		}

		void build(std::string &str) const;
		std::string build() const {
			std::string str;
			build(str);
			return str;
		}
	};
}

}

#endif
