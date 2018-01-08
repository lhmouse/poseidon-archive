// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_STREAM_BUFFER_HPP_
#define POSEIDON_STREAM_BUFFER_HPP_

#include "cxx_ver.hpp"
#include <string>
#include <utility>
#include <iterator>
#include <iosfwd>
#include <cstring>
#include <cstddef>

namespace Poseidon {

class StreamBuffer {
private:
	struct ChunkHeader;

public:
	class EnumerationCookie {
	public:
		ChunkHeader *prev;

	private:
		EnumerationCookie(const EnumerationCookie &);
		EnumerationCookie &operator=(const EnumerationCookie &);

	public:
		CONSTEXPR EnumerationCookie() NOEXCEPT
			: prev(NULLPTR)
		{ }
	};

	class ReadIterator;
	class WriteIterator;

private:
	ChunkHeader *m_first;
	ChunkHeader *m_last;
	std::size_t m_size;

public:
	CONSTEXPR StreamBuffer() NOEXCEPT
		: m_first(NULLPTR), m_last(NULLPTR), m_size(0)
	{ }
	StreamBuffer(const void *data, std::size_t count);
	explicit StreamBuffer(const char *str);
	explicit StreamBuffer(const std::string &str);
	explicit StreamBuffer(const std::basic_string<unsigned char> &str);
	StreamBuffer(const StreamBuffer &rhs);
	StreamBuffer &operator=(const StreamBuffer &rhs){
		StreamBuffer(rhs).swap(*this);
		return *this;
	}
#ifdef POSEIDON_CXX11
	StreamBuffer(StreamBuffer &&rhs) noexcept
		: StreamBuffer()
	{
		rhs.swap(*this);
	}
	StreamBuffer &operator=(StreamBuffer &&rhs) noexcept {
		rhs.swap(*this);
		return *this;
	}
#endif
	~StreamBuffer();

public:
	bool empty() const NOEXCEPT {
		return m_size == 0;
	}
	std::size_t size() const NOEXCEPT {
		return m_size;
	}
	void clear() NOEXCEPT;

	int front() const NOEXCEPT;
	int peek() const NOEXCEPT {
		return front();
	}
	int get() NOEXCEPT;
	bool discard() NOEXCEPT;
	void put(int data);
	int back() const NOEXCEPT;
	int unput() NOEXCEPT;
	void unget(int data);

	std::size_t peek(void *data, std::size_t count) const NOEXCEPT;
	std::size_t get(void *data, std::size_t count) NOEXCEPT;
	std::size_t discard(std::size_t count) NOEXCEPT;
	void put(int data, std::size_t count);
	void put(const void *data, std::size_t count);
	void put(const StreamBuffer &data);
	void put(const char *str){
		put(str, std::strlen(str));
	}
	void put(const std::string &str){
		put(str.data(), str.size());
	}
	void put(const std::basic_string<unsigned char> &str){
		put(str.data(), str.size());
	}

	void *squash();

	StreamBuffer cut_off(std::size_t count);
	void splice(StreamBuffer &rhs) NOEXCEPT;
#ifdef POSEIDON_CXX11
	void splice(StreamBuffer &&rhs) NOEXCEPT {
		splice(rhs);
	}
#endif

	bool enumerate_chunk(const void **data, std::size_t *count, EnumerationCookie &cookie) const NOEXCEPT;
	bool enumerate_chunk(void **data, std::size_t *count, EnumerationCookie &cookie) NOEXCEPT;

	void swap(StreamBuffer &rhs) NOEXCEPT {
		using std::swap;
		swap(m_first, rhs.m_first);
		swap(m_last, rhs.m_last);
		swap(m_size, rhs.m_size);
	}

	std::string dump_string() const;
	std::basic_string<unsigned char> dump_byte_string() const;
	void dump(std::ostream &os) const;

public:
	typedef unsigned char value_type;

	// std::front_insert_iterator
	void push_front(unsigned char by){
		unget(by);
	}

	// std::back_insert_iterator
	void push_back(unsigned char by){
		put(by);
	}
};

inline void swap(StreamBuffer &lhs, StreamBuffer &rhs) NOEXCEPT {
	lhs.swap(rhs);
}

extern std::ostream &operator<<(std::ostream &os, const StreamBuffer &rhs);

class StreamBuffer::ReadIterator : public std::iterator<std::input_iterator_tag, int> {
private:
	StreamBuffer *m_parent;

public:
	explicit ReadIterator(StreamBuffer &parent)
		: m_parent(&parent)
	{ }

public:
	int operator*() const {
		return m_parent->peek();
	}
	ReadIterator &operator++(){
		m_parent->discard();
		return *this;
	}
	ReadIterator &operator++(int){
		m_parent->discard();
		return *this;
	}
};

class StreamBuffer::WriteIterator : public std::iterator<std::output_iterator_tag, unsigned char> {
private:
	StreamBuffer *m_parent;

public:
	explicit WriteIterator(StreamBuffer &parent)
		: m_parent(&parent)
	{ }

public:
	WriteIterator &operator=(unsigned char byte){
		m_parent->put(byte);
		return *this;
	}
	WriteIterator &operator*(){
		return *this;
	}
	WriteIterator &operator++(){
		return *this;
	}
	WriteIterator &operator++(int){
		return *this;
	}
};

}

#endif
