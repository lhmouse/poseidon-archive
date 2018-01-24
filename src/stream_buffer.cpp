// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "stream_buffer.hpp"
#include "checked_arithmetic.hpp"
#include <boost/type_traits/common_type.hpp>

namespace Poseidon {

namespace {
	// XXX: Emulate C++14 `std::exchange()`.
	template<typename T>
	inline T exchange(T &t, typename boost::common_type<T>::type u){
		AUTO(v, STD_MOVE(t));
		t = STD_MOVE(u);
		return v;
	}
}

struct StreamBuffer::ChunkHeader {
	static ChunkHeader *create(std::size_t min_capacity, ChunkHeader *prev, ChunkHeader *next, bool backward){
		const std::size_t capacity = min_capacity | 1024;
		const std::size_t origin = backward ? capacity : 0;
		const AUTO(chunk, static_cast<ChunkHeader *>(::operator new(checked_add(sizeof(ChunkHeader), capacity))));
		chunk->capacity = capacity;
		chunk->prev = prev;
		chunk->next = next;
		chunk->begin = origin;
		chunk->end = origin;
		return chunk;
	}
	static void destroy(ChunkHeader *chunk) NOEXCEPT {
		::operator delete(chunk);
	}

	std::size_t capacity;
	ChunkHeader *prev;
	ChunkHeader *next;

	std::size_t begin;
	std::size_t end;
	__extension__ unsigned char data[];
};

StreamBuffer::StreamBuffer(const void *data, std::size_t count)
	: m_first(NULLPTR), m_last(NULLPTR), m_size(0)
{
	put(data, count);
}
StreamBuffer::StreamBuffer(const char *str)
	: m_first(NULLPTR), m_last(NULLPTR), m_size(0)
{
	put(str);
}
StreamBuffer::StreamBuffer(const std::string &str)
	: m_first(NULLPTR), m_last(NULLPTR), m_size(0)
{
	put(str);
}
StreamBuffer::StreamBuffer(const std::basic_string<unsigned char> &str)
	: m_first(NULLPTR), m_last(NULLPTR), m_size(0)
{
	put(str);
}
StreamBuffer::StreamBuffer(const StreamBuffer &rhs)
	: m_first(NULLPTR), m_last(NULLPTR), m_size(0)
{
	put(rhs);
}
StreamBuffer::~StreamBuffer(){
	AUTO(chunk, m_first);
	while(chunk){
		const AUTO(next, chunk->next);
		ChunkHeader::destroy(chunk);
		chunk = next;
	}
#ifndef NDEBUG
	std::memset(&m_first, 0xFE, sizeof(m_first));
	std::memset(&m_last,  0xFC, sizeof(m_last));
	std::memset(&m_size,  0xFA, sizeof(m_size));
#endif
}

void StreamBuffer::clear() NOEXCEPT {
	AUTO(chunk, m_first);
	while(chunk){
		chunk->begin = chunk->end;
		const AUTO(next, chunk->next);
		chunk = next;
	}
	m_size = 0;
}

int StreamBuffer::front() const NOEXCEPT {
	int read = -1;
	AUTO(chunk, m_first);
	while(chunk){
		if(chunk->end != chunk->begin){
			read = chunk->data[chunk->begin];
			break;
		}
		const AUTO(next, chunk->next);
		chunk = next;
	}
	return read;
}
int StreamBuffer::get() NOEXCEPT {
	int read = -1;
	AUTO(chunk, m_first);
	while(chunk){
		if(chunk->end != chunk->begin){
			read = chunk->data[chunk->begin];
			chunk->begin += 1;
			m_size -= 1;
			break;
		}
		const AUTO(next, chunk->next);
		(next ? next->prev : m_last) = NULLPTR;
		m_first = next;
		ChunkHeader::destroy(chunk);
		chunk = next;
	}
	return read;
}
bool StreamBuffer::discard() NOEXCEPT {
	bool discarded = false;
	AUTO(chunk, m_first);
	while(chunk){
		if(chunk->end != chunk->begin){
			discarded = true;
			chunk->begin += 1;
			m_size -= 1;
			break;
		}
		const AUTO(next, chunk->next);
		(next ? next->prev : m_last) = NULLPTR;
		m_first = next;
		ChunkHeader::destroy(chunk);
		chunk = next;
	}
	return discarded;
}
void StreamBuffer::put(int data){
	AUTO(chunk, m_last);
	AUTO(prev, chunk);
	if(chunk && (chunk->capacity == chunk->end)){
		const std::size_t avail = chunk->end - chunk->begin;
		if(chunk->capacity > avail){
			std::memmove(chunk->data, chunk->data + chunk->begin, avail);
			chunk->begin = 0;
			chunk->end = avail;
		} else {
			chunk = NULLPTR;
		}
	}
	if(!chunk){
		const AUTO(next, ChunkHeader::create(1, prev, NULLPTR, false));
		(prev ? prev->next : m_first) = next;
		m_last = next;
		chunk = next;
	}
	chunk->data[chunk->end] = static_cast<unsigned char>(data);
	chunk->end += 1;
	m_size += 1;
}
int StreamBuffer::back() const NOEXCEPT {
	int read = -1;
	AUTO(chunk, m_last);
	while(chunk){
		if(chunk->end != chunk->begin){
			read = chunk->data[chunk->end - 1];
			break;
		}
		const AUTO(prev, chunk->prev);
		chunk = prev;
	}
	return read;
}
int StreamBuffer::unput() NOEXCEPT {
	int read = -1;
	AUTO(chunk, m_last);
	while(chunk){
		if(chunk->end != chunk->begin){
			read = chunk->data[chunk->end - 1];
			chunk->end -= 1;
			m_size -= 1;
			break;
		}
		const AUTO(prev, chunk->prev);
		(prev ? prev->next : m_first) = NULLPTR;
		m_last = prev;
		ChunkHeader::destroy(chunk);
		chunk = prev;
	}
	return read;
}
void StreamBuffer::unget(int data){
	AUTO(chunk, m_first);
	AUTO(next, chunk);
	if(chunk && (chunk->begin == 0)){
		const std::size_t avail = chunk->end - chunk->begin;
		if(chunk->capacity > avail){
			std::memmove(chunk->data + chunk->begin + (chunk->capacity - chunk->end), chunk->data + chunk->begin, avail);
			chunk->begin = chunk->capacity - avail;
			chunk->end = chunk->capacity;
		} else {
			chunk = NULLPTR;
		}
	}
	if(!chunk){
		const AUTO(prev, ChunkHeader::create(1, NULLPTR, next, true));
		(next ? next->prev : m_last) = prev;
		m_first = prev;
		chunk = prev;
	}
	chunk->data[chunk->begin - 1] = static_cast<unsigned char>(data);
	chunk->begin -= 1;
	m_size += 1;
}

std::size_t StreamBuffer::peek(void *data, std::size_t count) const NOEXCEPT {
	std::size_t total = 0;
	AUTO(chunk, m_first);
	while(chunk){
		const std::size_t remaining = count - total;
		if(remaining == 0){
			break;
		}
		const std::size_t avail = chunk->end - chunk->begin;
		if(avail >= remaining){
			std::memcpy(static_cast<unsigned char *>(data) + total, chunk->data + chunk->begin, remaining);
			total += remaining;
			break;
		}
		std::memcpy(static_cast<unsigned char *>(data) + total, chunk->data + chunk->begin, avail);
		total += avail;
		const AUTO(next, chunk->next);
		chunk = next;
	}
	return total;
}
std::size_t StreamBuffer::get(void *data, std::size_t count) NOEXCEPT {
	std::size_t total = 0;
	AUTO(chunk, m_first);
	while(chunk){
		const std::size_t remaining = count - total;
		if(remaining == 0){
			break;
		}
		const std::size_t avail = chunk->end - chunk->begin;
		if(avail >= remaining){
			std::memcpy(static_cast<unsigned char *>(data) + total, chunk->data + chunk->begin, remaining);
			chunk->begin += remaining;
			m_size -= remaining;
			total += remaining;
			break;
		}
		std::memcpy(static_cast<unsigned char *>(data) + total, chunk->data + chunk->begin, avail);
		chunk->begin += avail;
		m_size -= avail;
		total += avail;
		const AUTO(next, chunk->next);
		(next ? next->prev : m_last) = NULLPTR;
		m_first = next;
		ChunkHeader::destroy(chunk);
		chunk = next;
	}
	return total;
}
std::size_t StreamBuffer::discard(std::size_t count) NOEXCEPT {
	std::size_t total = 0;
	AUTO(chunk, m_first);
	while(chunk){
		const std::size_t remaining = count - total;
		if(remaining == 0){
			break;
		}
		const std::size_t avail = chunk->end - chunk->begin;
		if(avail >= remaining){
			chunk->begin += remaining;
			m_size -= remaining;
			total += remaining;
			break;
		}
		chunk->begin += avail;
		m_size -= avail;
		total += avail;
		const AUTO(next, chunk->next);
		(next ? next->prev : m_last) = NULLPTR;
		m_first = next;
		ChunkHeader::destroy(chunk);
		chunk = next;
	}
	return total;
}
void StreamBuffer::put(int data, std::size_t count){
	AUTO(chunk, m_last);
	AUTO(prev, chunk);
	if(chunk && (chunk->capacity - chunk->end < count)){
		const std::size_t avail = chunk->end - chunk->begin;
		if(chunk->capacity - avail >= count){
			std::memmove(chunk->data, chunk->data + chunk->begin, avail);
			chunk->begin = 0;
			chunk->end = avail;
		} else {
			chunk = NULLPTR;
		}
	}
	if(!chunk){
		const AUTO(next, ChunkHeader::create(count, prev, NULLPTR, false));
		(prev ? prev->next : m_first) = next;
		chunk = next;
		m_last = next;
	}
	std::memset(chunk->data + chunk->end, data, count);
	chunk->end += count;
	m_size += count;
}
void StreamBuffer::put(const void *data, std::size_t count){
	AUTO(chunk, m_last);
	AUTO(prev, chunk);
	if(chunk && (chunk->capacity - chunk->end < count)){
		const std::size_t avail = chunk->end - chunk->begin;
		if(chunk->capacity - avail >= count){
			std::memmove(chunk->data, chunk->data + chunk->begin, avail);
			chunk->begin = 0;
			chunk->end = avail;
		} else {
			chunk = NULLPTR;
		}
	}
	if(!chunk){
		const AUTO(next, ChunkHeader::create(count, prev, NULLPTR, false));
		(prev ? prev->next : m_first) = next;
		chunk = next;
		m_last = next;
	}
	std::memcpy(chunk->data + chunk->end, data, count);
	chunk->end += count;
	m_size += count;
}
void StreamBuffer::put(const StreamBuffer &data){
	const AUTO(count, data.size());
	AUTO(chunk, m_last);
	AUTO(prev, chunk);
	if(chunk && (chunk->capacity - chunk->end < count)){
		const std::size_t avail = chunk->end - chunk->begin;
		if(chunk->capacity - avail >= count){
			std::memmove(chunk->data, chunk->data + chunk->begin, avail);
			chunk->begin = 0;
			chunk->end = avail;
		} else {
			chunk = NULLPTR;
		}
	}
	if(!chunk){
		const AUTO(next, ChunkHeader::create(count, prev, NULLPTR, false));
		(prev ? prev->next : m_first) = next;
		chunk = next;
		m_last = next;
	}
	for(AUTO(src, data.m_first); src; src = src->next){
		const std::size_t avail = src->end - src->begin;
		std::memcpy(chunk->data + chunk->end, src->data + src->begin, avail);
		chunk->end += avail;
	}
	m_size += count;
}

void *StreamBuffer::squash(){
	AUTO(chunk, m_first);
	if(!chunk){
		return NULLPTR;
	}
	if(chunk != m_last){
		StreamBuffer(*this).swap(*this);
		chunk = m_first;
	}
	return chunk->data + chunk->begin;
}

StreamBuffer StreamBuffer::cut_off(std::size_t count){
	std::size_t total = 0;
	AUTO(chunk, m_first);
	while(chunk){
		const std::size_t remaining = count - total;
		if(remaining == 0){
			break;
		}
		const std::size_t avail = chunk->end - chunk->begin;
		if(avail >= remaining){
			if(avail > remaining){
				const AUTO(prev, chunk->prev);
				const AUTO(next, chunk);
				chunk = ChunkHeader::create(remaining, prev, next, false);
				std::memcpy(chunk->data, next->data + next->begin, remaining);
				chunk->end = remaining;
				next->begin += remaining;
				(prev ? prev->next : m_first) = chunk;
				next->prev = chunk;
			}
			total += remaining;
			chunk = chunk->next;
			break;
		}
		total += avail;
		const AUTO(next, chunk->next);
		chunk = next;
	}
	StreamBuffer head;
	if(total != 0){
		const AUTO(first_cut, exchange(m_first, chunk));
		const AUTO(last_cut, exchange(chunk ? chunk->prev : m_last, NULLPTR));
		last_cut->next = NULLPTR;
		m_size -= total;
		head.m_first = first_cut;
		head.m_last = last_cut;
		head.m_size = total;
	}
	return head;
}
void StreamBuffer::splice(StreamBuffer &rhs) NOEXCEPT {
	assert(&rhs != this);

	const AUTO(first_add, exchange(rhs.m_first, NULLPTR));
	if(!first_add){
		return;
	}
	const AUTO(last, exchange(m_last, exchange(rhs.m_last, NULLPTR)));
	(last ? last->next : m_first) = first_add;
	first_add->prev = last;
	m_size += exchange(rhs.m_size, 0);
}

bool StreamBuffer::enumerate_chunk(const void **data, std::size_t *count, StreamBuffer::EnumerationCookie &cookie) const NOEXCEPT {
	const AUTO(chunk, cookie.prev ? cookie.prev->next : m_first);
	cookie.prev = chunk;
	if(!chunk){
		return false;
	}
	if(data){
		*data = chunk->data + chunk->begin;
	}
	if(count){
		*count = chunk->end - chunk->begin;
	}
	return true;
}
bool StreamBuffer::enumerate_chunk(void **data, std::size_t *count, StreamBuffer::EnumerationCookie &cookie) NOEXCEPT {
	const AUTO(chunk, cookie.prev ? cookie.prev->next : m_first);
	cookie.prev = chunk;
	if(!chunk){
		return false;
	}
	if(data){
		*data = chunk->data + chunk->begin;
	}
	if(count){
		*count = chunk->end - chunk->begin;
	}
	return true;
}

std::string StreamBuffer::dump_string() const {
	std::string str;
	str.reserve(m_size);
	AUTO(chunk, m_first);
	while(chunk){
		const std::size_t avail = chunk->end - chunk->begin;
		str.append(reinterpret_cast<const char *>(chunk->data + chunk->begin), avail);
		chunk = chunk->next;
	}
	return str;
}
std::basic_string<unsigned char> StreamBuffer::dump_byte_string() const {
	std::basic_string<unsigned char> str;
	str.reserve(m_size);
	AUTO(chunk, m_first);
	while(chunk){
		const std::size_t avail = chunk->end - chunk->begin;
		str.append(chunk->data + chunk->begin, avail);
		chunk = chunk->next;
	}
	return str;
}
void StreamBuffer::dump(std::ostream &os) const {
	AUTO(chunk, m_first);
	while(chunk){
		const std::size_t avail = chunk->end - chunk->begin;
		os.write(reinterpret_cast<const char *>(chunk->data + chunk->begin), static_cast<std::streamsize>(avail));
		chunk = chunk->next;
	}
}

}
