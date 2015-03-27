// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_STREAM_BUFFER_HPP_
#define POSEIDON_STREAM_BUFFER_HPP_

#include "cxx_ver.hpp"
#include <list>
#include <string>
#include <iosfwd>
#include <iterator>
#include <cstddef>

namespace Poseidon {

class StreamBuffer {
private:
	class Chunk;

public:
	class ReadIterator
		: public std::iterator<std::input_iterator_tag, int>
	{
	private:
		StreamBuffer *m_owner;

	public:
		explicit ReadIterator(StreamBuffer &owner)
			: m_owner(&owner)
		{
		}

	public:
		int operator*() const {
			return m_owner->peek();
		}
		ReadIterator &operator++(){
			m_owner->get();
			return *this;
		}
		ReadIterator operator++(int){
			m_owner->get();
			return *this;
		}
	};

	class WriteIterator
		: public std::iterator<std::output_iterator_tag, unsigned char>
	{
	private:
		StreamBuffer *m_owner;

	public:
		explicit WriteIterator(StreamBuffer &owner)
			: m_owner(&owner)
		{
		}

	public:
		WriteIterator &operator=(unsigned char byte){
			m_owner->put(byte);
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

private:
	std::list<Chunk> m_chunks;
	std::size_t m_size;

public:
	StreamBuffer();
	StreamBuffer(const void *data, std::size_t size);
	explicit StreamBuffer(const char *str);
	explicit StreamBuffer(const std::string &str);
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

	// 返回头部的一个字节。如果为空返回 -1，无异常抛出。
	int peek() const;
	// 返回头部的一个字节并删除之。如果为空返回 -1，无异常抛出。
	int get();
	// 向末尾追加一个字节。
	void put(unsigned char by);
	// put() 的逆操作。如果为空返回 -1，无异常抛出。
	int unput();
	// get() 的逆操作。
	void unget(unsigned char by);

	// 返回值是 data 中写入的字节数并且不大于 size。
	std::size_t peek(void *data, std::size_t size) const;
	std::size_t get(void *data, std::size_t size);
	std::size_t discard(std::size_t size);
	// 向末尾追加指定的字节数。
	void put(const void *data, std::size_t size);
	void put(const char *str);
	void put(const std::string &str);

	void swap(StreamBuffer &rhs) NOEXCEPT;

	// 拆分成两部分，返回 [0, size) 部分，[size, -) 部分仍保存于当前对象中。
	StreamBuffer cut(std::size_t size);
	// cut() 的逆操作。该函数返回后 src 为空。
	void splice(StreamBuffer &src) NOEXCEPT;
#ifdef POSEIDON_CXX11
	void splice(StreamBuffer &&src) noexcept {
		splice(src);
	}
#endif
	std::size_t transferFrom(StreamBuffer &src, std::size_t size){
		StreamBuffer tmp(src.cut(size));
		const std::size_t ret = tmp.size();
		splice(tmp);
		return ret;
	}
	std::size_t transferTo(StreamBuffer &dst, std::size_t size){
		return dst.transferFrom(*this, size);
	}

	void dump(std::string &str) const;
	void dump(std::ostream &os) const;
	void load(const std::string &str);
	void load(std::istream &is);

	void dumpHex(std::ostream &os) const;
};

inline void swap(StreamBuffer &lhs, StreamBuffer &rhs) NOEXCEPT {
	lhs.swap(rhs);
}

struct StreamBufferHexDumper {
	const StreamBuffer &buffer;

	explicit StreamBufferHexDumper(const StreamBuffer &buffer_)
		: buffer(buffer_)
	{
	}
};

inline std::ostream &operator<<(std::ostream &os, const StreamBufferHexDumper &rhs){
	rhs.buffer.dumpHex(os);
	return os;
}

}

#endif
