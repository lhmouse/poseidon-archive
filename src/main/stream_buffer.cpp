#include "../precompiled.hpp"
#include "stream_buffer.hpp"
#include <cassert>
#include <boost/thread/mutex.hpp>
using namespace Poseidon;

struct Poseidon::ImplDisposableBuffer {
	std::size_t readPos;
	std::size_t writePos;
	unsigned char data[1024];

	// 不初始化。如果我们不写构造函数这个结构会当成聚合，
	// 因此在 list 中会被 value-initialize 即填零，然而这是没有任何意义的。
	ImplDisposableBuffer(){
	}
};

namespace {

boost::mutex g_poolMutex;
std::list<ImplDisposableBuffer> g_pool;

ImplDisposableBuffer &pushBackPooled(std::list<ImplDisposableBuffer> &dst){
	do {
		{
			const boost::mutex::scoped_lock lock(g_poolMutex);
			if(!g_pool.empty()){
				dst.splice(dst.end(), g_pool, g_pool.begin());
				break;
			}
		}
		std::list<ImplDisposableBuffer> temp(1);
		dst.splice(dst.end(), temp, temp.begin());
	} while(false);

	AUTO_REF(ret, dst.back());
	ret.readPos = 0;
	ret.writePos = 0;
	return ret;
}
void popFrontPooled(std::list<ImplDisposableBuffer> &src){
	assert(!src.empty());

	const boost::mutex::scoped_lock lock(g_poolMutex);
	g_pool.splice(g_pool.begin(), src, src.begin());
}
void clearPooled(std::list<ImplDisposableBuffer> &src){
	if(src.empty()){
		return;
	}

	const boost::mutex::scoped_lock lock(g_poolMutex);
	g_pool.splice(g_pool.begin(), src);
}

}

StreamBuffer::StreamBuffer()
	: m_size(0)
{
}
StreamBuffer::StreamBuffer(const void *data, std::size_t size)
	: m_size(0)
{
	put(data, size);
}
StreamBuffer::StreamBuffer(const StreamBuffer &rhs)
	: m_chunks(rhs.m_chunks), m_size(rhs.m_size)
{
}
StreamBuffer &StreamBuffer::operator=(const StreamBuffer &rhs){
	StreamBuffer(rhs).swap(*this);
	return *this;
}
StreamBuffer::~StreamBuffer(){
	clear();
}

void StreamBuffer::clear(){
	clearPooled(m_chunks);
	m_size = 0;
}

int StreamBuffer::peek() const {
	if(m_size == 0){
		return -1;
	}
	AUTO(it, m_chunks.begin());
	for(;;){
		assert(it != m_chunks.end());

		if(it->readPos < it->writePos){
			return it->data[it->readPos];
		}
		++it;
	}
}
int StreamBuffer::get(){
	if(m_size == 0){
		return -1;
	}
	for(;;){
		assert(!m_chunks.empty());

		AUTO_REF(front, m_chunks.front());
		if(front.readPos < front.writePos){
			const int ret = front.data[front.readPos];
			++(front.readPos);
			--m_size;
			return ret;
		}
		popFrontPooled(m_chunks);
	}
}
void StreamBuffer::put(unsigned char by){
	AUTO(rit, m_chunks.rbegin());
	if((rit != m_chunks.rend()) && (rit->writePos < sizeof(rit->data))){
		rit->data[rit->writePos] = by;
		++(rit->writePos);
		++m_size;
		return;
	}

	AUTO_REF(back, pushBackPooled(m_chunks));
	back.data[back.writePos] = by;
	++(back.writePos);
	++m_size;
}

std::size_t StreamBuffer::peek(void *data, std::size_t size) const {
	if(m_size == 0){
		return 0;
	}

	const std::size_t ret = std::min(m_size, size);
	unsigned char *write = (unsigned char *)data;
	std::size_t copied = 0;
	AUTO(it, m_chunks.begin());
	for(;;){
		assert(it != m_chunks.end());

		const std::size_t toCopy = std::min(ret - copied, it->writePos - it->readPos);
		std::memcpy(write, it->data + it->readPos, toCopy);
		write += toCopy;
		copied += toCopy;
		if(copied == ret){
			break;
		}
		++it;
	}
	return ret;
}
std::size_t StreamBuffer::get(void *data, std::size_t size){
	if(m_size == 0){
		return 0;
	}

	const std::size_t ret = std::min(m_size, size);
	unsigned char *write = (unsigned char *)data;
	std::size_t copied = 0;
	for(;;){
		assert(!m_chunks.empty());

		AUTO_REF(front, m_chunks.front());
		const std::size_t toCopy = std::min(ret - copied, front.writePos - front.readPos);
		std::memcpy(write, front.data + front.readPos, toCopy);
		write += toCopy;
		front.readPos += toCopy;
		m_size -= toCopy;
		copied += toCopy;
		if(copied == ret){
			break;
		}
		popFrontPooled(m_chunks);
	}
	return ret;
}
std::size_t StreamBuffer::discard(std::size_t size){
	if(m_size == 0){
		return 0;
	}

	const std::size_t ret = std::min(m_size, size);
	std::size_t dropped = 0;
	for(;;){
		assert(!m_chunks.empty());

		AUTO_REF(front, m_chunks.front());
		const std::size_t toDrop = std::min(ret - dropped, front.writePos - front.readPos);
		front.readPos += toDrop;
		m_size -= toDrop;
		dropped += toDrop;
		if(dropped == ret){
			break;
		}
		popFrontPooled(m_chunks);
	}
	return ret;
}
void StreamBuffer::put(const void *data, std::size_t size){
	const unsigned char *read = (const unsigned char *)data;
	std::size_t copied = 0;

	AUTO(rit, m_chunks.rbegin());
	if((rit != m_chunks.rend()) && (rit->writePos < sizeof(rit->data))){
		const std::size_t toCopy = std::min(size - copied, sizeof(rit->data) - rit->writePos);
		std::memcpy(rit->data + rit->writePos, read, toCopy);
		rit->writePos += toCopy;
		read += toCopy;
		m_size += toCopy;
		copied += toCopy;
	}
	while(copied < size){
		AUTO_REF(back, pushBackPooled(m_chunks));
		const std::size_t toCopy = std::min(size - copied, sizeof(back.data));
		std::memcpy(back.data, read, toCopy);
		back.writePos = toCopy;
		read += toCopy;
		m_size += toCopy;
		copied += toCopy;
	}
}

void StreamBuffer::swap(StreamBuffer &rhs){
	m_chunks.swap(rhs.m_chunks);
	std::swap(m_size, rhs.m_size);
}

StreamBuffer StreamBuffer::cut(std::size_t size){
	StreamBuffer ret;
	if(m_size <= size){
		ret.swap(*this);
	} else {
		AUTO(it, m_chunks.begin());
		std::size_t total = 0; // 这是 [m_chunks.begin(), it) 的字节数，不含零头。
		while(total < size){
			assert(it != m_chunks.end());

			const std::size_t remaining = size - total;
			const std::size_t avail = it->writePos - it->readPos;
			if(remaining < avail){
				pushBackPooled(ret.m_chunks);
				AUTO_REF(back, ret.m_chunks.back());
				std::memcpy(back.data, it->data + it->readPos, remaining);
				back.writePos = remaining;
				it->readPos += remaining;
				ret.m_size += remaining;
				m_size -= remaining;
				break;
			}
			total += avail;
			++it;
		}
		ret.m_chunks.splice(ret.m_chunks.begin(), m_chunks, m_chunks.begin(), it);
		ret.m_size += total;
		m_size -= total;
	}
	return ret;
}
void StreamBuffer::splice(StreamBuffer &src){
	if(&src == this){
		return;
	}
	m_chunks.splice(m_chunks.end(), src.m_chunks);
	m_size += src.m_size;
	src.m_size = 0;
}
