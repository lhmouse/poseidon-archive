#ifndef POSEIDON_STREAM_BUFFER_HPP_
#define POSEIDON_STREAM_BUFFER_HPP_

#include "../cxx_ver.hpp"
#include <list>
#include <string>
#include <iosfwd>
#include <iterator>
#include <cstddef>

namespace Poseidon {

struct ImplDisposableBuffer;

class StreamBuffer {
private:
	std::list<ImplDisposableBuffer> m_chunks;
	std::size_t m_size;

public:
	StreamBuffer();
	StreamBuffer(const void *data, std::size_t size);
	StreamBuffer(const char *str);
	StreamBuffer(const StreamBuffer &rhs);
	StreamBuffer &operator=(const StreamBuffer &rhs);
#ifdef POSEIDON_CXX11
	StreamBuffer(StreamBuffer &&rhs) noexcept;
	StreamBuffer &operator=(StreamBuffer &&rhs) noexcept;
#endif
	~StreamBuffer();

public:
	bool empty() const {
		return m_size == 0;
	}
	std::size_t size() const {
		return m_size;
	}
	void clear();

	// 返回头部的一个字节。如果为空返回 -1。
	int peek() const;
	int get();
	// 向末尾追加一个字节。
	void put(unsigned char by);

	// 返回值是 data 中写入的字节数并且不大于 size。
	std::size_t peek(void *data, std::size_t size) const;
	std::size_t get(void *data, std::size_t size);
	std::size_t discard(std::size_t size);
	// 向末尾追加指定的字节数。
	void put(const void *data, std::size_t size);
	void put(const char *str);

	void swap(StreamBuffer &rhs) NOEXCEPT;

	// 拆分成两部分，返回 [0, size) 部分，[size, -) 部分仍保存于当前对象中。
	StreamBuffer cut(std::size_t size);
	// cut() 的逆操作。该函数返回后 src 为空。
	void splice(StreamBuffer &src) NOEXCEPT;
#ifdef POSEIDON_CXX11
	void splice(StreamBuffer &&src) noexcept {
		splice(std::move(src));
	}
#endif

	std::string dump() const;
	void dump(std::ostream &os) const;
	void load(const std::string &str);
	void load(std::istream &is);
};

class StreamBufferReadIterator
	: public std::iterator<std::input_iterator_tag, int>
{
private:
	StreamBuffer *m_owner;

public:
	explicit StreamBufferReadIterator(StreamBuffer &owner)
		: m_owner(&owner)
	{
	}

public:
	int operator*() const {
		return m_owner->peek();
	}
	StreamBufferReadIterator &operator++(){
		m_owner->get();
		return *this;
	}
	StreamBufferReadIterator operator++(int){
		m_owner->get();
		return *this;
	}
};

class StreamBufferWriteIterator
	: public std::iterator<std::output_iterator_tag, unsigned char>
{
private:
	StreamBuffer *m_owner;

public:
	explicit StreamBufferWriteIterator(StreamBuffer &owner)
		: m_owner(&owner)
	{
	}

public:
	StreamBufferWriteIterator &operator=(unsigned char byte){
		m_owner->put(byte);
		return *this;
	}
	StreamBufferWriteIterator &operator*(){
		return *this;
	}
	StreamBufferWriteIterator &operator++(){
		return *this;
	}
	StreamBufferWriteIterator &operator++(int){
		return *this;
	}
};

}

#endif
