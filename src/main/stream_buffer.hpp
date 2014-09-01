#ifndef POSEIDON_STREAM_BUFFER_HPP_
#define POSEIDON_STREAM_BUFFER_HPP_

#include "../cxxver.hpp"
#include <list>
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
	StreamBuffer(const StreamBuffer &rhs);
	StreamBuffer &operator=(const StreamBuffer &rhs);
#ifdef POSEIDON_CXX11
	StreamBuffer(StreamBuffer &&rhs) noexcept;
	StreamBuffer &operator=(StreamBuffer &&rhs) noexcept;
#endif
	~StreamBuffer() throw();


public:
	bool empty() const throw() {
		return m_size == 0;
	}
	std::size_t size() const throw() {
		return m_size;
	}
	void clear() throw();

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

	void swap(StreamBuffer &rhs) throw();

	// 拆分成两部分，返回 [0, size) 部分，[size, -) 部分仍保存于当前对象中。
	StreamBuffer cut(std::size_t size);
	// cut() 的逆操作。该函数返回后 src 为空。
	void splice(StreamBuffer &src) throw();
};

}

#endif
