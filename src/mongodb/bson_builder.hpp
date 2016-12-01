// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_MONGODB_BSON_BUILDER_HPP_
#define POSEIDON_MONGODB_BSON_BUILDER_HPP_

#include "../cxx_ver.hpp"
#include <boost/container/deque.hpp>
#include <boost/cstdint.hpp>
#include <string>
#include <iosfwd>
#include <cstddef>
#include "../shared_nts.hpp"

namespace Poseidon {

class Uuid;

namespace MongoDb {
	class BsonBuilder {
	private:
		enum Type {
			T_BOOLEAN    =  1,
			T_SIGNED     =  2,
			T_UNSIGNED   =  3,
			T_DOUBLE     =  4,
			T_STRING     =  5,
			T_DATETIME   =  6,
			T_UUID       =  7,
			T_BLOB       =  8,

			T_JS_CODE    = 93,
			T_REGEX      = 94,
			T_MINKEY     = 95,
			T_MAXKEY     = 96,
			T_NULL       = 97,
			T_OBJECT     = 98,
			T_ARRAY      = 99,
		};

		struct Element {
			SharedNts name;
			Type type;
			std::string large;
			unsigned char small[16];
		};

	private:
		boost::container::deque<Element> m_elements;

	public:
		BsonBuilder()
			: m_elements()
		{
		}
#ifndef POSEIDON_CXX11
		BsonBuilder(const BsonBuilder &rhs)
			: m_elements(rhs.m_elements)
		{
		}
		BsonBuilder &operator=(const BsonBuilder &rhs){
			m_elements = rhs.m_elements;
			return *this;
		}
#endif

	private:
		void internal_build(void *impl, bool as_array) const;

	public:
		void append_boolean(SharedNts name, bool value);
		void append_signed(SharedNts name, boost::int64_t value);
		void append_unsigned(SharedNts name, boost::uint64_t value);
		void append_double(SharedNts name, double value);
		void append_string(SharedNts name, std::string value);
		void append_datetime(SharedNts name, boost::uint64_t value);
		void append_uuid(SharedNts name, const Uuid &value);
		void append_blob(SharedNts name, std::string value);

		void append_js_code(SharedNts name, std::string code);
		void append_regex(SharedNts name, std::string regex, const char *options = "");
		void append_minkey(SharedNts name);
		void append_maxkey(SharedNts name);
		void append_null(SharedNts name);
		void append_object(SharedNts name, const BsonBuilder &obj);
		void append_array(SharedNts name, const BsonBuilder &arr);

		bool empty() const {
			return m_elements.empty();
		}
		std::size_t size() const {
			return m_elements.size();
		}
		void clear() NOEXCEPT {
			m_elements.clear();
		}

		void swap(BsonBuilder &rhs) NOEXCEPT {
			using std::swap;
			swap(m_elements, rhs.m_elements);
		}

		std::string build(bool as_array = false) const;
		void build(std::ostream &os, bool as_array = false) const;

		std::string build_json(bool as_array = false) const;
		void build_json(std::ostream &os, bool as_array = false) const;
	};

	inline void swap(BsonBuilder &lhs, BsonBuilder &rhs) NOEXCEPT {
		lhs.swap(rhs);
	}

	inline std::ostream &operator<<(std::ostream &os, const BsonBuilder &rhs){
		rhs.build_json(os);
		return os;
	}

	inline BsonBuilder bson_scalar_boolean(SharedNts name, bool value){
		BsonBuilder ret;
		ret.append_boolean(STD_MOVE(name), value);
		return ret;
	}
	inline BsonBuilder bson_scalar_signed(SharedNts name, boost::int64_t value){
		BsonBuilder ret;
		ret.append_signed(STD_MOVE(name), value);
		return ret;
	}
	inline BsonBuilder bson_scalar_unsigned(SharedNts name, boost::uint64_t value){
		BsonBuilder ret;
		ret.append_unsigned(STD_MOVE(name), value);
		return ret;
	}
	inline BsonBuilder bson_scalar_double(SharedNts name, double value){
		BsonBuilder ret;
		ret.append_double(STD_MOVE(name), value);
		return ret;
	}
	inline BsonBuilder bson_scalar_string(SharedNts name, std::string value){
		BsonBuilder ret;
		ret.append_string(STD_MOVE(name), STD_MOVE(value));
		return ret;
	}
	inline BsonBuilder bson_scalar_datetime(SharedNts name, boost::uint64_t value){
		BsonBuilder ret;
		ret.append_datetime(STD_MOVE(name), value);
		return ret;
	}
	inline BsonBuilder bson_scalar_uuid(SharedNts name, const Uuid &value){
		BsonBuilder ret;
		ret.append_uuid(STD_MOVE(name), value);
		return ret;
	}
	inline BsonBuilder bson_scalar_blob(SharedNts name, std::string value){
		BsonBuilder ret;
		ret.append_blob(STD_MOVE(name), STD_MOVE(value));
		return ret;
	}

	inline BsonBuilder bson_scalar_regex(SharedNts name, std::string regex, const char *options = ""){
		BsonBuilder ret;
		ret.append_regex(STD_MOVE(name), STD_MOVE(regex), options);
		return ret;
	}
	inline BsonBuilder bson_scalar_minkey(SharedNts name){
		BsonBuilder ret;
		ret.append_minkey(STD_MOVE(name));
		return ret;
	}
	inline BsonBuilder bson_scalar_maxkey(SharedNts name){
		BsonBuilder ret;
		ret.append_maxkey(STD_MOVE(name));
		return ret;
	}
	inline BsonBuilder bson_scalar_null(SharedNts name){
		BsonBuilder ret;
		ret.append_null(STD_MOVE(name));
		return ret;
	}
	inline BsonBuilder bson_scalar_object(SharedNts name, const BsonBuilder &obj){
		BsonBuilder ret;
		ret.append_object(STD_MOVE(name), obj);
		return ret;
	}
	inline BsonBuilder bson_scalar_array(SharedNts name, const BsonBuilder &arr){
		BsonBuilder ret;
		ret.append_array(STD_MOVE(name), arr);
		return ret;
	}
}

}

#endif
