// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "bson_builder.hpp"
#include "../exception.hpp"
#include "../time.hpp"
#include "../profiler.hpp"
#include "../buffer_streams.hpp"
#include "../raii.hpp"
#include <libbson-1.0/bson.h>

namespace Poseidon {
namespace Mongodb {

namespace {
	struct Bson_closer {
		CONSTEXPR ::bson_t * operator()() const NOEXCEPT {
			return NULLPTR;
		}
		void operator()(::bson_t *bt) const NOEXCEPT {
			::bson_destroy(bt);
		}
	};
	struct Bson_string_deleter {
		CONSTEXPR char * operator()() const NOEXCEPT {
			return NULLPTR;
		}
		void operator()(char *str) const NOEXCEPT {
			::bson_free(str);
		}
	};
}

void Bson_builder::internal_build(void *impl, bool as_array) const {
	POSEIDON_PROFILE_ME;

	const AUTO(bt, static_cast< ::bson_t *>(impl));

	for(AUTO(it, m_elements.begin()); it != m_elements.end(); ++it){
		char key_storage[32];
		const char *key_str;
		if(as_array){
			::bson_uint32_to_string(static_cast<std::uint32_t>(it - m_elements.begin()), &key_str, key_storage, sizeof(key_storage));
		} else {
			key_str = it->name.get();
		}
		switch(it->type){
		case type_boolean: {
			bool value;
			std::memcpy(&value, it->small, sizeof(value));
			POSEIDON_THROW_UNLESS(::bson_append_bool(bt, key_str, -1, value), Basic_exception, Rcnts::view("BSON builder: bson_append_bool() failed"));
			break; }
		case type_signed: {
			std::int64_t value;
			std::memcpy(&value, it->small, sizeof(value));
			POSEIDON_THROW_UNLESS(::bson_append_int64(bt, key_str, -1, value), Basic_exception, Rcnts::view("BSON builder: bson_append_int64() failed"));
			break; }
		case type_unsigned: {
			std::uint64_t value;
			std::memcpy(&value, it->small, sizeof(value));
			POSEIDON_THROW_UNLESS(::bson_append_int64(bt, key_str, -1, boost::numeric_cast<std::int64_t>(value)), Basic_exception, Rcnts::view("BSON builder: bson_append_int64() failed"));
			break; }
		case type_double: {
			double value;
			std::memcpy(&value, it->small, sizeof(value));
			POSEIDON_THROW_UNLESS(::bson_append_double(bt, key_str, -1, value), Basic_exception, Rcnts::view("BSON builder: bson_append_double() failed"));
			break; }
		case type_string: {
			POSEIDON_THROW_UNLESS(::bson_append_utf8(bt, key_str, -1, it->large.data(), boost::numeric_cast<int>(it->large.size())), Basic_exception, Rcnts::view("BSON builder: bson_append_utf8() failed"));
			break; }
		case type_datetime: {
			std::uint64_t value;
			std::memcpy(&value, it->small, sizeof(value));
			char str[64];
			std::size_t len = format_time(str, sizeof(str), value, true);
			POSEIDON_THROW_UNLESS(::bson_append_utf8(bt, key_str, -1, str, static_cast<int>(len)), Basic_exception, Rcnts::view("BSON builder: bson_append_utf8() failed"));
			break; }
		case type_uuid: {
			char str[36];
			Uuid(it->small).to_string(str);
			POSEIDON_THROW_UNLESS(::bson_append_utf8(bt, key_str, -1, str, sizeof(str)), Basic_exception, Rcnts::view("BSON builder: bson_append_utf8() failed"));
			break; }
		case type_blob: {
			POSEIDON_THROW_UNLESS(::bson_append_binary(bt, key_str, -1, BSON_SUBTYPE_BINARY,
				reinterpret_cast<const std::uint8_t *>(it->large.data()), boost::numeric_cast<unsigned>(it->large.size())), Basic_exception, Rcnts::view("BSON builder: bson_append_binary() failed"));
			break; }
		case type_js_code: {
			POSEIDON_THROW_UNLESS(::bson_append_code(bt, key_str, -1, it->large.c_str()), Basic_exception, Rcnts::view("BSON builder: bson_append_code() failed"));
			break; }
		case type_regex: {
			POSEIDON_THROW_UNLESS(::bson_append_regex(bt, key_str, -1, it->large.c_str(), it->small), Basic_exception, Rcnts::view("BSON builder: bson_append_regex() failed"));
			break; }
		case type_minkey: {
			POSEIDON_THROW_UNLESS(::bson_append_minkey(bt, key_str, -1), Basic_exception, Rcnts::view("BSON builder: bson_append_minkey() failed"));
			break; }
		case type_maxkey: {
			POSEIDON_THROW_UNLESS(::bson_append_maxkey(bt, key_str, -1), Basic_exception, Rcnts::view("BSON builder: bson_append_maxkey() failed"));
			break; }
		case type_null: {
			POSEIDON_THROW_UNLESS(::bson_append_null(bt, key_str, -1), Basic_exception, Rcnts::view("BSON builder: bson_append_null() failed"));
			break; }
		case type_object: {
			::bson_t child_storage;
			POSEIDON_THROW_UNLESS(::bson_init_static(&child_storage, reinterpret_cast<const std::uint8_t *>(it->large.data()), it->large.size()), Basic_exception, Rcnts::view("BSON builder: bson_init_static() failed"));
			const Unique_handle<Bson_closer> child_guard(&child_storage);
			const AUTO(child_bt, child_guard.get());
			POSEIDON_THROW_UNLESS(::bson_append_document(bt, key_str, -1, child_bt), Basic_exception, Rcnts::view("BSON builder: bson_append_document() failed"));
			break; }
		case type_array: {
			::bson_t child_storage;
			POSEIDON_THROW_UNLESS(::bson_init_static(&child_storage, reinterpret_cast<const std::uint8_t *>(it->large.data()), it->large.size()), Basic_exception, Rcnts::view("BSON builder: bson_init_static() failed"));
			const Unique_handle<Bson_closer> child_guard(&child_storage);
			const AUTO(child_bt, child_guard.get());
			POSEIDON_THROW_UNLESS(::bson_append_array(bt, key_str, -1, child_bt), Basic_exception, Rcnts::view("BSON builder: bson_append_array() failed"));
			break; }
		default:
			POSEIDON_THROW(Basic_exception, Rcnts::view("BSON builder: Unknown element type"));
		}
	}
}

void Bson_builder::append_boolean(Rcnts name, bool value){
	Element elem = { STD_MOVE(name), type_boolean };
	BOOST_STATIC_ASSERT(sizeof(elem.small) >= sizeof(value));
	std::memcpy(elem.small, &value, sizeof(value));
	m_elements.push_back(STD_MOVE(elem));
}
void Bson_builder::append_signed(Rcnts name, std::int64_t value){
	Element elem = { STD_MOVE(name), type_signed };
	BOOST_STATIC_ASSERT(sizeof(elem.small) >= sizeof(value));
	std::memcpy(elem.small, &value, sizeof(value));
	m_elements.push_back(STD_MOVE(elem));
}
void Bson_builder::append_unsigned(Rcnts name, std::uint64_t value){
	Element elem = { STD_MOVE(name), type_unsigned };
	BOOST_STATIC_ASSERT(sizeof(elem.small) >= sizeof(value));
	std::memcpy(elem.small, &value, sizeof(value));
	m_elements.push_back(STD_MOVE(elem));
}
void Bson_builder::append_double(Rcnts name, double value){
	Element elem = { STD_MOVE(name), type_double };
	BOOST_STATIC_ASSERT(sizeof(elem.small) >= sizeof(value));
	std::memcpy(elem.small, &value, sizeof(value));
	m_elements.push_back(STD_MOVE(elem));
}
void Bson_builder::append_string(Rcnts name, const std::string &value){
	Element elem = { STD_MOVE(name), type_string };
	elem.large = value;
	m_elements.push_back(STD_MOVE(elem));
}
void Bson_builder::append_datetime(Rcnts name, std::uint64_t value){
	Element elem = { STD_MOVE(name), type_datetime };
	BOOST_STATIC_ASSERT(sizeof(elem.small) >= sizeof(value));
	std::memcpy(elem.small, &value, sizeof(value));
	m_elements.push_back(STD_MOVE(elem));
}
void Bson_builder::append_uuid(Rcnts name, const Uuid &value){
	Element elem = { STD_MOVE(name), type_uuid };
	BOOST_STATIC_ASSERT(sizeof(elem.small) >= sizeof(value));
	std::memcpy(elem.small, value.data(), value.size());
	m_elements.push_back(STD_MOVE(elem));
}
void Bson_builder::append_blob(Rcnts name, const Stream_buffer &value){
	Element elem = { STD_MOVE(name), type_blob };
	elem.large = value.dump_string();
	m_elements.push_back(STD_MOVE(elem));
}

void Bson_builder::append_js_code(Rcnts name, const std::string &code){
	Element elem = { STD_MOVE(name), type_js_code };
	elem.large = code;
	m_elements.push_back(STD_MOVE(elem));
}
void Bson_builder::append_regex(Rcnts name, const std::string &regex, const char *options){
	Element elem = { STD_MOVE(name), type_regex };
	if(options){
		::stpncpy(elem.small, options, sizeof(elem.small) - 1)[0] = 0;
	} else {
		elem.small[0] = 0;
	}
	elem.large = regex;
	m_elements.push_back(STD_MOVE(elem));
}
void Bson_builder::append_minkey(Rcnts name){
	Element elem = { STD_MOVE(name), type_minkey };
	m_elements.push_back(STD_MOVE(elem));
}
void Bson_builder::append_maxkey(Rcnts name){
	Element elem = { STD_MOVE(name), type_maxkey };
	m_elements.push_back(STD_MOVE(elem));
}
void Bson_builder::append_null(Rcnts name){
	Element elem = { STD_MOVE(name), type_null };
	m_elements.push_back(STD_MOVE(elem));
}
void Bson_builder::append_object(Rcnts name, const Bson_builder &obj){
	Element elem = { STD_MOVE(name), type_object };
	Buffer_ostream os;
	obj.build(os, false);
	elem.large = os.get_buffer().dump_string();
	m_elements.push_back(STD_MOVE(elem));
}
void Bson_builder::append_array(Rcnts name, const Bson_builder &arr){
	Element elem = { STD_MOVE(name), type_array };
	Buffer_ostream os;
	arr.build(os, true);
	elem.large = os.get_buffer().dump_string();
	m_elements.push_back(STD_MOVE(elem));
}

Stream_buffer Bson_builder::build(bool as_array) const {
	POSEIDON_PROFILE_ME;

	Buffer_ostream os;
	build(os, as_array);
	return STD_MOVE(os.get_buffer());
}
void Bson_builder::build(std::ostream &os, bool as_array) const {
	POSEIDON_PROFILE_ME;

	::bson_t bt_storage = BSON_INITIALIZER;
	const Unique_handle<Bson_closer> bt_guard(&bt_storage);
	const AUTO(bt, bt_guard.get());

	internal_build(bt, as_array);

	os.write(reinterpret_cast<const char *>(::bson_get_data(bt)), boost::numeric_cast<std::streamsize>(bt->len));
}

std::string Bson_builder::build_json(bool as_array) const {
	POSEIDON_PROFILE_ME;

	Buffer_ostream os;
	build_json(os, as_array);
	return os.get_buffer().dump_string();
}
void Bson_builder::build_json(std::ostream &os, bool as_array) const {
	POSEIDON_PROFILE_ME;

	::bson_t bt_storage = BSON_INITIALIZER;
	const Unique_handle<Bson_closer> bt_guard(&bt_storage);
	const AUTO(bt, bt_guard.get());

	internal_build(bt, as_array);

	const AUTO(json, ::bson_as_json(bt, NULLPTR));
	POSEIDON_THROW_UNLESS(json, Basic_exception, Rcnts::view("BSON builder: Failed to convert BSON to JSON"));
	const Unique_handle<Bson_string_deleter> json_guard(json);

	os <<json;
}

}
}
