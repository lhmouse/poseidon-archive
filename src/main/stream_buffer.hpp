#ifndef POSEIDON_STREAM_BUFFER_HPP_
#define POSEIDON_STREAM_BUFFER_HPP_

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
	StreamBuffer(const StreamBuffer &rhs);
	StreamBuffer &operator=(const StreamBuffer &rhs);
	~StreamBuffer();

public:
	bool empty() const {
		return m_size != 0;
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
	// 向末尾追加指定的字节数。
	void put(const void *data, std::size_t size);

	void swap(StreamBuffer &rhs);
};

}

#endif
