// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_MONGODB_BSON_BUILDER_HPP_
#define POSEIDON_MONGODB_BSON_BUILDER_HPP_

#include "../cxx_ver.hpp"
#include "../shared_nts.hpp"
#include "../uuid.hpp"
#include <boost/container/deque.hpp>
#include <boost/cstdint.hpp>
#include <iosfwd>
#include <cstddef>

namespace Poseidon {
namespace Mongodb {

class Bson_builder {
private:
	enum Type {
		type_boolean    =  1,
		type_signed     =  2,
		type_unsigned   =  3,
		type_double     =  4,
		type_string     =  5,
		type_datetime   =  6,
		type_uuid       =  7,
		type_blob       =  8,

		type_js_code    = 93,
		type_regex      = 94,
		type_minkey     = 95,
		type_maxkey     = 96,
		type_null       = 97,
		type_object     = 98,
		type_array      = 99,
	};

	struct Element {
		Shared_nts name;
		Type type;
		std::string large;
		char small[16];
	};

private:
	boost::container::deque<Element> m_elements;

public:
	Bson_builder()
		: m_elements()
	{
		//
	}
#ifndef POSEIDON_CXX11
	Bson_builder(const Bson_builder &rhs)
		: m_elements(rhs.m_elements)
	{
		//
	}
	Bson_builder &operator=(const Bson_builder &rhs){
		m_elements = rhs.m_elements;
		return *this;
	}
#endif

private:
	void internal_build(void *impl, bool as_array) const;

public:
	void append_boolean(Shared_nts name, bool value);
	void append_signed(Shared_nts name, boost::int64_t value);
	void append_unsigned(Shared_nts name, boost::uint64_t value);
	void append_double(Shared_nts name, double value);
	void append_string(Shared_nts name, const std::string &value);
	void append_datetime(Shared_nts name, boost::uint64_t value);
	void append_uuid(Shared_nts name, const Uuid &value);
	void append_blob(Shared_nts name, const std::basic_string<unsigned char> &value);

	void append_js_code(Shared_nts name, const std::string &code);
	void append_regex(Shared_nts name, const std::string &regex, const char *options = "");
	void append_minkey(Shared_nts name);
	void append_maxkey(Shared_nts name);
	void append_null(Shared_nts name);
	void append_object(Shared_nts name, const Bson_builder &obj);
	void append_array(Shared_nts name, const Bson_builder &arr);

	bool empty() const {
		return m_elements.empty();
	}
	std::size_t size() const {
		return m_elements.size();
	}
	void clear() NOEXCEPT {
		m_elements.clear();
	}

	void swap(Bson_builder &rhs) NOEXCEPT {
		using std::swap;
		swap(m_elements, rhs.m_elements);
	}

	std::basic_string<unsigned char> build(bool as_array = false) const;
	void build(std::ostream &os, bool as_array = false) const;

	std::string build_json(bool as_array = false) const;
	void build_json(std::ostream &os, bool as_array = false) const;
};

inline void swap(Bson_builder &lhs, Bson_builder &rhs) NOEXCEPT {
	lhs.swap(rhs);
}

inline std::ostream &operator<<(std::ostream &os, const Bson_builder &rhs){
	rhs.build_json(os);
	return os;
}

inline Bson_builder bson_scalar_boolean(Shared_nts name, bool value){
	Bson_builder ret;
	ret.append_boolean(STD_MOVE(name), value);
	return ret;
}
inline Bson_builder bson_scalar_signed(Shared_nts name, boost::int64_t value){
	Bson_builder ret;
	ret.append_signed(STD_MOVE(name), value);
	return ret;
}
inline Bson_builder bson_scalar_unsigned(Shared_nts name, boost::uint64_t value){
	Bson_builder ret;
	ret.append_unsigned(STD_MOVE(name), value);
	return ret;
}
inline Bson_builder bson_scalar_double(Shared_nts name, double value){
	Bson_builder ret;
	ret.append_double(STD_MOVE(name), value);
	return ret;
}
inline Bson_builder bson_scalar_string(Shared_nts name, const std::string &value){
	Bson_builder ret;
	ret.append_string(STD_MOVE(name), value);
	return ret;
}
inline Bson_builder bson_scalar_datetime(Shared_nts name, boost::uint64_t value){
	Bson_builder ret;
	ret.append_datetime(STD_MOVE(name), value);
	return ret;
}
inline Bson_builder bson_scalar_uuid(Shared_nts name, const Uuid &value){
	Bson_builder ret;
	ret.append_uuid(STD_MOVE(name), value);
	return ret;
}
inline Bson_builder bson_scalar_blob(Shared_nts name, const std::basic_string<unsigned char> &value){
	Bson_builder ret;
	ret.append_blob(STD_MOVE(name), value);
	return ret;
}

inline Bson_builder bson_scalar_regex(Shared_nts name, const std::string &regex, const char *options = ""){
	Bson_builder ret;
	ret.append_regex(STD_MOVE(name), regex, options);
	return ret;
}
inline Bson_builder bson_scalar_minkey(Shared_nts name){
	Bson_builder ret;
	ret.append_minkey(STD_MOVE(name));
	return ret;
}
inline Bson_builder bson_scalar_maxkey(Shared_nts name){
	Bson_builder ret;
	ret.append_maxkey(STD_MOVE(name));
	return ret;
}
inline Bson_builder bson_scalar_null(Shared_nts name){
	Bson_builder ret;
	ret.append_null(STD_MOVE(name));
	return ret;
}
inline Bson_builder bson_scalar_object(Shared_nts name, const Bson_builder &obj){
	Bson_builder ret;
	ret.append_object(STD_MOVE(name), obj);
	return ret;
}
inline Bson_builder bson_scalar_array(Shared_nts name, const Bson_builder &arr){
	Bson_builder ret;
	ret.append_array(STD_MOVE(name), arr);
	return ret;
}

}
}

#endif
