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

class Stream_buffer {
private:
	struct Chunk_header;

public:
	class Enumeration_cookie;
	class Read_iterator;
	class Write_iterator;

private:
	Chunk_header *m_first;
	Chunk_header *m_last;
	std::size_t m_size;

public:
	CONSTEXPR Stream_buffer() NOEXCEPT
		: m_first(NULLPTR), m_last(NULLPTR), m_size(0)
	{
		//
	}
	Stream_buffer(const void *data, std::size_t count);
	explicit Stream_buffer(const char *str);
	explicit Stream_buffer(const std::string &str);
	explicit Stream_buffer(const std::basic_string<unsigned char> &str);
	Stream_buffer(const Stream_buffer &rhs);
	Stream_buffer &operator=(const Stream_buffer &rhs){
		Stream_buffer(rhs).swap(*this);
		return *this;
	}
#ifdef POSEIDON_CXX11
	Stream_buffer(Stream_buffer &&rhs) noexcept
		: Stream_buffer()
	{
		rhs.swap(*this);
	}
	Stream_buffer &operator=(Stream_buffer &&rhs) noexcept {
		rhs.swap(*this);
		return *this;
	}
#endif
	~Stream_buffer();

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
	void put(const Stream_buffer &data);
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

	Stream_buffer cut_off(std::size_t count);
	void splice(Stream_buffer &rhs) NOEXCEPT;
#ifdef POSEIDON_CXX11
	void splice(Stream_buffer &&rhs) noexcept {
		splice(rhs);
	}
#endif

	bool enumerate_chunk(const void **data, std::size_t *count, Enumeration_cookie &cookie) const NOEXCEPT;
	bool enumerate_chunk(void **data, std::size_t *count, Enumeration_cookie &cookie) NOEXCEPT;

	void swap(Stream_buffer &rhs) NOEXCEPT {
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

inline void swap(Stream_buffer &lhs, Stream_buffer &rhs) NOEXCEPT {
	lhs.swap(rhs);
}

inline std::ostream &operator<<(std::ostream &os, const Stream_buffer &rhs){
	rhs.dump(os);
	return os;
}

class Stream_buffer::Enumeration_cookie {
	friend Stream_buffer;

private:
	Chunk_header *m_prev;

public:
	CONSTEXPR Enumeration_cookie() NOEXCEPT
		: m_prev(NULLPTR)
	{
		//
	}
};


class Stream_buffer::Read_iterator : public std::iterator<std::input_iterator_tag, int> {
private:
	Stream_buffer *m_parent;
	mutable int m_byte;

public:
	explicit Read_iterator(Stream_buffer &parent)
		: m_parent(&parent)
	{
		//
	}

public:
	const int & operator*() const {
		m_byte = m_parent->peek();
		return m_byte;
	}
	Read_iterator &operator++(){
		m_parent->discard();
		return *this;
	}
	Read_iterator &operator++(int){
		m_parent->discard();
		return *this;
	}
};

class Stream_buffer::Write_iterator : public std::iterator<std::output_iterator_tag, unsigned char> {
private:
	Stream_buffer *m_parent;

public:
	explicit Write_iterator(Stream_buffer &parent)
		: m_parent(&parent)
	{
		//
	}

public:
	Write_iterator &operator=(unsigned char byte){
		m_parent->put(byte);
		return *this;
	}
	Write_iterator &operator*(){
		return *this;
	}
	Write_iterator &operator++(){
		return *this;
	}
	Write_iterator &operator++(int){
		return *this;
	}
};

}

#endif
