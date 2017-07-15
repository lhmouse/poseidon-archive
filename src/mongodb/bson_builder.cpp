// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "bson_builder.hpp"
#include "../protocol_exception.hpp"
#include "../time.hpp"
#include "../uuid.hpp"
#include "../profiler.hpp"
#include "../buffer_streams.hpp"
#include "../raii.hpp"
#pragma GCC push_options
#pragma GCC diagnostic ignored "-Wsign-conversion"
#include <bson.h>
#pragma GCC pop_options

namespace Poseidon {

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

	inline int narrowing_cast_to_int(std::size_t size){
		if(size > INT_MAX){
			DEBUG_THROW(ProtocolException, sslit("BSON builder: The value is too large to fit into an int"), -1);
		}
		return static_cast<int>(size);
	}
}

namespace MongoDb {
	void BsonBuilder::internal_build(void *impl, bool as_array) const {
		PROFILE_ME;

		const AUTO(bt, static_cast< ::bson_t *>(impl));

		for(AUTO(it, m_elements.begin()); it != m_elements.end(); ++it){
			char key_storage[32];
			const char *key_str;
			if(as_array){
				::bson_uint32_to_string(static_cast<boost::uint32_t>(it - m_elements.begin()),
					&key_str, key_storage, sizeof(key_storage));
			} else {
				key_str = it->name.get();
			}
			switch(it->type){
			case T_BOOLEAN: {
				bool value;
				std::memcpy(&value, it->small, sizeof(value));
				if(!::bson_append_bool(bt, key_str, -1, value)){
					DEBUG_THROW(ProtocolException, sslit("BSON builder: bson_append_bool() failed"), -1);
				}
				break; }
			case T_SIGNED: {
				boost::int64_t value;
				std::memcpy(&value, it->small, sizeof(value));
				if(!::bson_append_int64(bt, key_str, -1, value)){
					DEBUG_THROW(ProtocolException, sslit("BSON builder: bson_append_int64() failed"), -1);
				}
				break; }
			case T_UNSIGNED: {
				boost::uint64_t value;
				std::memcpy(&value, it->small, sizeof(value));
				boost::int64_t shifted = static_cast<boost::int64_t>(value - (1ull << 63));
				if(!::bson_append_int64(bt, key_str, -1, shifted)){
					DEBUG_THROW(ProtocolException, sslit("BSON builder: bson_append_int64() failed"), -1);
				}
				break; }
			case T_DOUBLE: {
				double value;
				std::memcpy(&value, it->small, sizeof(value));
				if(!::bson_append_double(bt, key_str, -1, value)){
					DEBUG_THROW(ProtocolException, sslit("BSON builder: bson_append_double() failed"), -1);
				}
				break; }
			case T_STRING: {
				if(!::bson_append_utf8(bt, key_str, -1, it->large.data(), narrowing_cast_to_int(it->large.size()))){
					DEBUG_THROW(ProtocolException, sslit("BSON builder: bson_append_utf8() failed"), -1);
				}
				break; }
			case T_DATETIME: {
				boost::uint64_t value;
				std::memcpy(&value, it->small, sizeof(value));
				char str[64];
				std::size_t len = format_time(str, sizeof(str), value, true);
				if(!::bson_append_utf8(bt, key_str, -1, str, (int)len)){
					DEBUG_THROW(ProtocolException, sslit("BSON builder: bson_append_utf8() failed"), -1);
				}
				break; }
			case T_UUID: {
				char str[36];
				Uuid(it->small).to_string(str);
				if(!::bson_append_utf8(bt, key_str, -1, str, sizeof(str))){
					DEBUG_THROW(ProtocolException, sslit("BSON builder: bson_append_utf8() failed"), -1);
				}
				break; }
			case T_BLOB: {
				if(!::bson_append_binary(bt, key_str, -1, BSON_SUBTYPE_BINARY,
					reinterpret_cast<const boost::uint8_t *>(it->large.data()),
					static_cast<unsigned>(narrowing_cast_to_int(it->large.size()))))
				{
					DEBUG_THROW(ProtocolException, sslit("BSON builder: bson_append_binary() failed"), -1);
				}
				break; }
			case T_JS_CODE: {
				if(!::bson_append_code(bt, key_str, -1, it->large.c_str())){
					DEBUG_THROW(ProtocolException, sslit("BSON builder: bson_append_code() failed"), -1);
				}
				break; }
			case T_REGEX: {
				if(!::bson_append_regex(bt, key_str, -1, it->large.c_str(), reinterpret_cast<const char *>(it->small))){
					DEBUG_THROW(ProtocolException, sslit("BSON builder: bson_append_regex() failed"), -1);
				}
				break; }
			case T_MINKEY: {
				if(!::bson_append_minkey(bt, key_str, -1)){
					DEBUG_THROW(ProtocolException, sslit("BSON builder: bson_append_minkey() failed"), -1);
				}
				break; }
			case T_MAXKEY: {
				if(!::bson_append_maxkey(bt, key_str, -1)){
					DEBUG_THROW(ProtocolException, sslit("BSON builder: bson_append_maxkey() failed"), -1);
				}
				break; }
			case T_NULL: {
				if(!::bson_append_null(bt, key_str, -1)){
					DEBUG_THROW(ProtocolException, sslit("BSON builder: bson_append_null() failed"), -1);
				}
				break; }
			case T_OBJECT: {
				::bson_t child_storage;
				if(!::bson_init_static(&child_storage,
					reinterpret_cast<const boost::uint8_t *>(it->large.data()), it->large.size()))
				{
					DEBUG_THROW(ProtocolException, sslit("BSON builder: bson_init_static() failed"), -1);
				}
				const UniqueHandle<BsonCloser> child_guard(&child_storage);
				const AUTO(child_bt, child_guard.get());
				if(!::bson_append_document(bt, key_str, -1, child_bt)){
					DEBUG_THROW(ProtocolException, sslit("BSON builder: bson_append_document() failed"), -1);
				}
				break; }
			case T_ARRAY: {
				::bson_t child_storage;
				if(!::bson_init_static(&child_storage,
					reinterpret_cast<const boost::uint8_t *>(it->large.data()), it->large.size()))
				{
					DEBUG_THROW(ProtocolException, sslit("BSON builder: bson_init_static() failed"), -1);
				}
				const UniqueHandle<BsonCloser> child_guard(&child_storage);
				const AUTO(child_bt, child_guard.get());
				if(!::bson_append_array(bt, key_str, -1, child_bt)){
					DEBUG_THROW(ProtocolException, sslit("BSON builder: bson_append_array() failed"), -1);
				}
				break; }
			default:
				DEBUG_THROW(ProtocolException, sslit("BSON builder: Unknown element type"), -1);
			}
		}
	}

	void BsonBuilder::append_boolean(SharedNts name, bool value){
		Element elem = { STD_MOVE(name), T_BOOLEAN };
		assert(sizeof(elem.small) >= sizeof(value));
		std::memcpy(elem.small, &value, sizeof(value));
		m_elements.push_back(STD_MOVE(elem));
	}
	void BsonBuilder::append_signed(SharedNts name, boost::int64_t value){
		Element elem = { STD_MOVE(name), T_SIGNED };
		assert(sizeof(elem.small) >= sizeof(value));
		std::memcpy(elem.small, &value, sizeof(value));
		m_elements.push_back(STD_MOVE(elem));
	}
	void BsonBuilder::append_unsigned(SharedNts name, boost::uint64_t value){
		Element elem = { STD_MOVE(name), T_UNSIGNED };
		assert(sizeof(elem.small) >= sizeof(value));
		std::memcpy(elem.small, &value, sizeof(value));
		m_elements.push_back(STD_MOVE(elem));
	}
	void BsonBuilder::append_double(SharedNts name, double value){
		Element elem = { STD_MOVE(name), T_DOUBLE };
		assert(sizeof(elem.small) >= sizeof(value));
		std::memcpy(elem.small, &value, sizeof(value));
		m_elements.push_back(STD_MOVE(elem));
	}
	void BsonBuilder::append_string(SharedNts name, std::string value){
		Element elem = { STD_MOVE(name), T_STRING };
		elem.large.swap(value);
		m_elements.push_back(STD_MOVE(elem));
	}
	void BsonBuilder::append_datetime(SharedNts name, boost::uint64_t value){
		Element elem = { STD_MOVE(name), T_DATETIME };
		assert(sizeof(elem.small) >= sizeof(value));
		std::memcpy(elem.small, &value, sizeof(value));
		m_elements.push_back(STD_MOVE(elem));
	}
	void BsonBuilder::append_uuid(SharedNts name, const Uuid &value){
		Element elem = { STD_MOVE(name), T_UUID };
		assert(sizeof(elem.small) >= value.size());
		std::memcpy(elem.small, value.data(), value.size());
		m_elements.push_back(STD_MOVE(elem));
	}
	void BsonBuilder::append_blob(SharedNts name, std::string value){
		Element elem = { STD_MOVE(name), T_BLOB };
		elem.large.swap(value);
		m_elements.push_back(STD_MOVE(elem));
	}

	void BsonBuilder::append_js_code(SharedNts name, std::string code){
		Element elem = { STD_MOVE(name), T_JS_CODE };
		elem.large.swap(code);
		m_elements.push_back(STD_MOVE(elem));
	}
	void BsonBuilder::append_regex(SharedNts name, std::string regex, const char *options){
		Element elem = { STD_MOVE(name), T_REGEX };
		if(options){
			const AUTO(len, std::min(std::strlen(options), sizeof(elem.small) - 1));
			std::memcpy(elem.small, options, len);
			elem.small[len] = 0;
		} else {
			elem.small[0] = 0;
		}
		elem.large.swap(regex);
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
		std::string bin = obj.build(false);
		elem.large.swap(bin);
		m_elements.push_back(STD_MOVE(elem));
	}
	void BsonBuilder::append_array(SharedNts name, const BsonBuilder &arr){
		Element elem = { STD_MOVE(name), T_ARRAY };
		std::string bin = arr.build(true);
		elem.large.swap(bin);
		m_elements.push_back(STD_MOVE(elem));
	}

	std::string BsonBuilder::build(bool as_array) const {
		PROFILE_ME;

		Buffer_ostream os;
		build(os, as_array);
		return os.get_buffer().dump_string();
	}
	void BsonBuilder::build(std::ostream &os, bool as_array) const {
		PROFILE_ME;

		::bson_t bt_storage = BSON_INITIALIZER;
		const UniqueHandle<BsonCloser> bt_guard(&bt_storage);
		const AUTO(bt, bt_guard.get());

		internal_build(bt, as_array);

		os.write(reinterpret_cast<const char *>(::bson_get_data(bt)), static_cast<std::streamsize>(bt->len));
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
		if(!json){
			DEBUG_THROW(ProtocolException, sslit("BSON builder: Failed to convert BSON to JSON"), -1);
		}
		const UniqueHandle<BsonStringDeleter> json_guard(json);

		os <<json;
	}
}

}
