// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "bson_builder.hpp"

#if __GNUC__ >= 6
#  pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

#include "../exception.hpp"
#include "../time.hpp"
#include "../profiler.hpp"
#include "../buffer_streams.hpp"
#include "../raii.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#include <libbson-1.0/bson.h>
#pragma GCC diagnostic pop

namespace Poseidon {
namespace MongoDb {

namespace {
	struct BsonCloser {
		CONSTEXPR ::bson_t *operator()() const NOEXCEPT {
			return NULLPTR;
		}
		void operator()(::bson_t *bt) const NOEXCEPT {
			::bson_destroy(bt);
		}
	};
	struct BsonStringDeleter {
		CONSTEXPR char *operator()() const NOEXCEPT {
			return NULLPTR;
		}
		void operator()(char *str) const NOEXCEPT {
			::bson_free(str);
		}
	};
}

void BsonBuilder::internal_build(void *impl, bool as_array) const {
	PROFILE_ME;

	const AUTO(bt, static_cast< ::bson_t *>(impl));

	for(AUTO(it, m_elements.begin()); it != m_elements.end(); ++it){
		char key_storage[32];
		const char *key_str;
		if(as_array){
			::bson_uint32_to_string(static_cast<boost::uint32_t>(it - m_elements.begin()), &key_str, key_storage, sizeof(key_storage));
		} else {
			key_str = it->name.get();
		}
		switch(it->type){
		case T_BOOLEAN: {
			bool value;
			std::memcpy(&value, it->small, sizeof(value));
			DEBUG_THROW_UNLESS(::bson_append_bool(bt, key_str, -1, value), BasicException, sslit("BSON builder: bson_append_bool() failed"));
			break; }
		case T_SIGNED: {
			boost::int64_t value;
			std::memcpy(&value, it->small, sizeof(value));
			DEBUG_THROW_UNLESS(::bson_append_int64(bt, key_str, -1, value), BasicException, sslit("BSON builder: bson_append_int64() failed"));
			break; }
		case T_UNSIGNED: {
			boost::uint64_t value;
			std::memcpy(&value, it->small, sizeof(value));
			boost::int64_t shifted = static_cast<boost::int64_t>(value - (1ull << 63));
			DEBUG_THROW_UNLESS(::bson_append_int64(bt, key_str, -1, shifted), BasicException, sslit("BSON builder: bson_append_int64() failed"));
			break; }
		case T_DOUBLE: {
			double value;
			std::memcpy(&value, it->small, sizeof(value));
			DEBUG_THROW_UNLESS(::bson_append_double(bt, key_str, -1, value), BasicException, sslit("BSON builder: bson_append_double() failed"));
			break; }
		case T_STRING: {
			DEBUG_THROW_UNLESS(::bson_append_utf8(bt, key_str, -1, it->large.data(), boost::numeric_cast<int>(it->large.size())), BasicException, sslit("BSON builder: bson_append_utf8() failed"));
			break; }
		case T_DATETIME: {
			boost::uint64_t value;
			std::memcpy(&value, it->small, sizeof(value));
			char str[64];
			std::size_t len = format_time(str, sizeof(str), value, true);
			DEBUG_THROW_UNLESS(::bson_append_utf8(bt, key_str, -1, str, static_cast<int>(len)), BasicException, sslit("BSON builder: bson_append_utf8() failed"));
			break; }
		case T_UUID: {
			char str[36];
			Uuid(it->small).to_string(str);
			DEBUG_THROW_UNLESS(::bson_append_utf8(bt, key_str, -1, str, sizeof(str)), BasicException, sslit("BSON builder: bson_append_utf8() failed"));
			break; }
		case T_BLOB: {
			DEBUG_THROW_UNLESS(::bson_append_binary(bt, key_str, -1, BSON_SUBTYPE_BINARY,
				reinterpret_cast<const boost::uint8_t *>(it->large.data()), boost::numeric_cast<unsigned>(it->large.size())), BasicException, sslit("BSON builder: bson_append_binary() failed"));
			break; }
		case T_JS_CODE: {
			DEBUG_THROW_UNLESS(::bson_append_code(bt, key_str, -1, it->large.c_str()), BasicException, sslit("BSON builder: bson_append_code() failed"));
			break; }
		case T_REGEX: {
			DEBUG_THROW_UNLESS(::bson_append_regex(bt, key_str, -1, it->large.c_str(), it->small), BasicException, sslit("BSON builder: bson_append_regex() failed"));
			break; }
		case T_MINKEY: {
			DEBUG_THROW_UNLESS(::bson_append_minkey(bt, key_str, -1), BasicException, sslit("BSON builder: bson_append_minkey() failed"));
			break; }
		case T_MAXKEY: {
			DEBUG_THROW_UNLESS(::bson_append_maxkey(bt, key_str, -1), BasicException, sslit("BSON builder: bson_append_maxkey() failed"));
			break; }
		case T_NULL: {
			DEBUG_THROW_UNLESS(::bson_append_null(bt, key_str, -1), BasicException, sslit("BSON builder: bson_append_null() failed"));
			break; }
		case T_OBJECT: {
			::bson_t child_storage;
			DEBUG_THROW_UNLESS(::bson_init_static(&child_storage, reinterpret_cast<const boost::uint8_t *>(it->large.data()), it->large.size()), BasicException, sslit("BSON builder: bson_init_static() failed"));
			const UniqueHandle<BsonCloser> child_guard(&child_storage);
			const AUTO(child_bt, child_guard.get());
			DEBUG_THROW_UNLESS(::bson_append_document(bt, key_str, -1, child_bt), BasicException, sslit("BSON builder: bson_append_document() failed"));
			break; }
		case T_ARRAY: {
			::bson_t child_storage;
			DEBUG_THROW_UNLESS(::bson_init_static(&child_storage, reinterpret_cast<const boost::uint8_t *>(it->large.data()), it->large.size()), BasicException, sslit("BSON builder: bson_init_static() failed"));
			const UniqueHandle<BsonCloser> child_guard(&child_storage);
			const AUTO(child_bt, child_guard.get());
			DEBUG_THROW_UNLESS(::bson_append_array(bt, key_str, -1, child_bt), BasicException, sslit("BSON builder: bson_append_array() failed"));
			break; }
		default:
			DEBUG_THROW(BasicException, sslit("BSON builder: Unknown element type"));
		}
	}
}

void BsonBuilder::append_boolean(SharedNts name, bool value){
	Element elem = { STD_MOVE(name), T_BOOLEAN };
	BOOST_STATIC_ASSERT(sizeof(elem.small) >= sizeof(value));
	std::memcpy(elem.small, &value, sizeof(value));
	m_elements.push_back(STD_MOVE(elem));
}
void BsonBuilder::append_signed(SharedNts name, boost::int64_t value){
	Element elem = { STD_MOVE(name), T_SIGNED };
	BOOST_STATIC_ASSERT(sizeof(elem.small) >= sizeof(value));
	std::memcpy(elem.small, &value, sizeof(value));
	m_elements.push_back(STD_MOVE(elem));
}
void BsonBuilder::append_unsigned(SharedNts name, boost::uint64_t value){
	Element elem = { STD_MOVE(name), T_UNSIGNED };
	BOOST_STATIC_ASSERT(sizeof(elem.small) >= sizeof(value));
	std::memcpy(elem.small, &value, sizeof(value));
	m_elements.push_back(STD_MOVE(elem));
}
void BsonBuilder::append_double(SharedNts name, double value){
	Element elem = { STD_MOVE(name), T_DOUBLE };
	BOOST_STATIC_ASSERT(sizeof(elem.small) >= sizeof(value));
	std::memcpy(elem.small, &value, sizeof(value));
	m_elements.push_back(STD_MOVE(elem));
}
void BsonBuilder::append_string(SharedNts name, const std::string &value){
	Element elem = { STD_MOVE(name), T_STRING };
	elem.large.append(value.data(), value.size());
	m_elements.push_back(STD_MOVE(elem));
}
void BsonBuilder::append_datetime(SharedNts name, boost::uint64_t value){
	Element elem = { STD_MOVE(name), T_DATETIME };
	BOOST_STATIC_ASSERT(sizeof(elem.small) >= sizeof(value));
	std::memcpy(elem.small, &value, sizeof(value));
	m_elements.push_back(STD_MOVE(elem));
}
void BsonBuilder::append_uuid(SharedNts name, const Uuid &value){
	Element elem = { STD_MOVE(name), T_UUID };
	BOOST_STATIC_ASSERT(sizeof(elem.small) >= sizeof(value));
	std::memcpy(elem.small, value.data(), value.size());
	m_elements.push_back(STD_MOVE(elem));
}
void BsonBuilder::append_blob(SharedNts name, const std::basic_string<unsigned char> &value){
	Element elem = { STD_MOVE(name), T_BLOB };
	elem.large.append(reinterpret_cast<const char *>(value.data()), value.size());
	m_elements.push_back(STD_MOVE(elem));
}

void BsonBuilder::append_js_code(SharedNts name, const std::string &code){
	Element elem = { STD_MOVE(name), T_JS_CODE };
	elem.large.append(code.data(), code.size());
	m_elements.push_back(STD_MOVE(elem));
}
void BsonBuilder::append_regex(SharedNts name, const std::string &regex, const char *options){
	Element elem = { STD_MOVE(name), T_REGEX };
	if(options){
		::stpncpy(elem.small, options, sizeof(elem.small) - 1)[0] = 0;
	} else {
		elem.small[0] = 0;
	}
	elem.large.append(regex.data(), regex.size());
	m_elements.push_back(STD_MOVE(elem));
}
void BsonBuilder::append_minkey(SharedNts name){
	Element elem = { STD_MOVE(name), T_MINKEY };
	m_elements.push_back(STD_MOVE(elem));
}
void BsonBuilder::append_maxkey(SharedNts name){
	Element elem = { STD_MOVE(name), T_MAXKEY };
	m_elements.push_back(STD_MOVE(elem));
}
void BsonBuilder::append_null(SharedNts name){
	Element elem = { STD_MOVE(name), T_NULL };
	m_elements.push_back(STD_MOVE(elem));
}
void BsonBuilder::append_object(SharedNts name, const BsonBuilder &obj){
	Element elem = { STD_MOVE(name), T_OBJECT };
	Buffer_ostream os;
	obj.build(os, false);
	elem.large = os.get_buffer().dump_string();
	m_elements.push_back(STD_MOVE(elem));
}
void BsonBuilder::append_array(SharedNts name, const BsonBuilder &arr){
	Element elem = { STD_MOVE(name), T_ARRAY };
	Buffer_ostream os;
	arr.build(os, true);
	elem.large = os.get_buffer().dump_string();
	m_elements.push_back(STD_MOVE(elem));
}

std::basic_string<unsigned char> BsonBuilder::build(bool as_array) const {
	PROFILE_ME;

	Buffer_ostream os;
	build(os, as_array);
	return os.get_buffer().dump_byte_string();
}
void BsonBuilder::build(std::ostream &os, bool as_array) const {
	PROFILE_ME;

	::bson_t bt_storage = BSON_INITIALIZER;
	const UniqueHandle<BsonCloser> bt_guard(&bt_storage);
	const AUTO(bt, bt_guard.get());

	internal_build(bt, as_array);

	os.write(reinterpret_cast<const char *>(::bson_get_data(bt)), boost::numeric_cast<std::streamsize>(bt->len));
}

std::string BsonBuilder::build_json(bool as_array) const {
	PROFILE_ME;

	Buffer_ostream os;
	build_json(os, as_array);
	return os.get_buffer().dump_string();
}
void BsonBuilder::build_json(std::ostream &os, bool as_array) const {
	PROFILE_ME;

	::bson_t bt_storage = BSON_INITIALIZER;
	const UniqueHandle<BsonCloser> bt_guard(&bt_storage);
	const AUTO(bt, bt_guard.get());

	internal_build(bt, as_array);

	const AUTO(json, ::bson_as_json(bt, NULLPTR));
	DEBUG_THROW_UNLESS(json, BasicException, sslit("BSON builder: Failed to convert BSON to JSON"));
	const UniqueHandle<BsonStringDeleter> json_guard(json);

	os <<json;
}

}
}
