// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "stream_buffer.hpp"
#include <iostream>
#include <cassert>
#include <boost/thread/mutex.hpp>
using namespace Poseidon;

class StreamBuffer::Chunk {
private:
	static boost::mutex s_mutex;
	static std::list<Chunk> s_pool;

public:
	static Chunk &pushBackPooled(std::list<Chunk> &dst){
		{
			const boost::mutex::scoped_lock lock(s_mutex);
			if(!s_pool.empty()){
				dst.splice(dst.end(), s_pool, s_pool.begin());
				goto done;
			}
		}
		dst.push_back(VAL_INIT);

	done:
		AUTO_REF(ret, dst.back());
		ret.m_readPos = 0;
		ret.m_writePos = 0;
		return ret;
	}
	static void popBackPooled(std::list<Chunk> &src){
		assert(!src.empty());

		AUTO(it, src.end());
		--it;
		const boost::mutex::scoped_lock lock(s_mutex);
		s_pool.splice(s_pool.begin(), src, it);
	}
	static Chunk &pushFrontPooled(std::list<Chunk> &dst){
		{
			const boost::mutex::scoped_lock lock(s_mutex);
			if(!s_pool.empty()){
				dst.splice(dst.begin(), s_pool, s_pool.begin());
				goto done;
			}
		}
		dst.push_front(VAL_INIT);

	done:
		AUTO_REF(ret, dst.back());
		ret.m_readPos = sizeof(ret.m_data);
		ret.m_writePos = sizeof(ret.m_data);
		return ret;
	}
	static void popFrontPooled(std::list<Chunk> &src){
		assert(!src.empty());

		const boost::mutex::scoped_lock lock(s_mutex);
		s_pool.splice(s_pool.begin(), src, src.begin());
	}
	static void clearPooled(std::list<Chunk> &src){
		if(src.empty()){
			return;
		}
		const boost::mutex::scoped_lock lock(s_mutex);
		s_pool.splice(s_pool.begin(), src);
	}

public:
	std::size_t m_readPos;
	std::size_t m_writePos;
	char m_data[256];

public:
	// 不初始化。如果我们不写构造函数这个结构会当成聚合，
	// 因此在 list 中会被 value-initialize 即填零，然而这是没有任何意义的。
	Chunk(){
	}
};

boost::mutex					StreamBuffer::Chunk::s_mutex	__attribute__((__init_priority__(500)));
std::list<StreamBuffer::Chunk>	StreamBuffer::Chunk::s_pool		__attribute__((__init_priority__(500)));

StreamBuffer::StreamBuffer()
	: m_size(0)
{
}
StreamBuffer::StreamBuffer(const void *data, std::size_t size)
	: m_size(0)
{
	put(data, size);
}
StreamBuffer::StreamBuffer(const char *str)
	: m_size(0)
{
	put(str);
}
StreamBuffer::StreamBuffer(const std::string &str)
	: m_size(0)
{
	put(str);
}
StreamBuffer::StreamBuffer(const StreamBuffer &rhs)
	: m_size(0)
{
	for(AUTO(it, rhs.m_chunks.begin()); it != rhs.m_chunks.end(); ++it){
		put(it->m_data + it->m_readPos, it->m_writePos - it->m_readPos);
	}
}
StreamBuffer &StreamBuffer::operator=(const StreamBuffer &rhs){
	StreamBuffer(rhs).swap(*this);
	return *this;
}
#ifdef POSEIDON_CXX11
StreamBuffer::StreamBuffer(StreamBuffer &&rhs) noexcept
	: m_size(0)
{
	swap(rhs);
}
StreamBuffer &StreamBuffer::operator=(StreamBuffer &&rhs) noexcept {
	rhs.swap(*this);
	return *this;
}
#endif
StreamBuffer::~StreamBuffer(){
	clear();
}

void StreamBuffer::clear(){
	Chunk::clearPooled(m_chunks);
	m_size = 0;
}

int StreamBuffer::peek() const {
	if(m_size == 0){
		return -1;
	}
	AUTO(it, m_chunks.begin());
	for(;;){
		assert(it != m_chunks.end());

		if(it->m_readPos < it->m_writePos){
			return it->m_data[it->m_readPos];
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
		if(front.m_readPos < front.m_writePos){
			const int ret = front.m_data[front.m_readPos];
			++front.m_readPos;
			--m_size;
			return ret;
		}
		Chunk::popFrontPooled(m_chunks);
	}
}
void StreamBuffer::put(unsigned char by){
	if(!m_chunks.empty()){
		AUTO_REF(back, m_chunks.back());
		if(back.m_writePos < sizeof(back.m_data)){
			back.m_data[back.m_writePos] = static_cast<char>(by);
			++back.m_writePos;
			++m_size;
			return;
		}
	}

	AUTO_REF(back, Chunk::pushBackPooled(m_chunks));
	back.m_data[back.m_writePos] = by;
	++back.m_writePos;
	++m_size;
}
int StreamBuffer::unput(){
	if(m_size == 0){
		return -1;
	}
	for(;;){
		assert(!m_chunks.empty());

		AUTO_REF(front, m_chunks.back());
		if(front.m_readPos < front.m_writePos){
			--front.m_writePos;
			const int ret = front.m_data[front.m_writePos];
			--m_size;
			return ret;
		}
		Chunk::popBackPooled(m_chunks);
	}
}
void StreamBuffer::unget(unsigned char by){
	if(!m_chunks.empty()){
		AUTO_REF(front, m_chunks.front());
		if(0 < front.m_readPos){
			++front.m_readPos;
			front.m_data[front.m_readPos] = static_cast<char>(by);
			++m_size;
			return;
		}
	}

	AUTO_REF(front, Chunk::pushFrontPooled(m_chunks));
	++front.m_readPos;
	front.m_data[front.m_readPos] = by;
	++m_size;
}

std::size_t StreamBuffer::peek(void *data, std::size_t size) const {
	if(m_size == 0){
		return 0;
	}

	const std::size_t ret = std::min(m_size, size);
	char *write = static_cast<char *>(data);
	std::size_t copied = 0;
	AUTO(it, m_chunks.begin());
	for(;;){
		assert(it != m_chunks.end());

		const std::size_t toCopy = std::min(ret - copied, it->m_writePos - it->m_readPos);
		std::memcpy(write, it->m_data + it->m_readPos, toCopy);
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
	char *write = static_cast<char *>(data);
	std::size_t copied = 0;
	for(;;){
		assert(!m_chunks.empty());

		AUTO_REF(front, m_chunks.front());
		const std::size_t toCopy = std::min(ret - copied, front.m_writePos - front.m_readPos);
		std::memcpy(write, front.m_data + front.m_readPos, toCopy);
		write += toCopy;
		front.m_readPos += toCopy;
		m_size -= toCopy;
		copied += toCopy;
		if(copied == ret){
			break;
		}
		Chunk::popFrontPooled(m_chunks);
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
		const std::size_t toDrop = std::min(ret - dropped, front.m_writePos - front.m_readPos);
		front.m_readPos += toDrop;
		m_size -= toDrop;
		dropped += toDrop;
		if(dropped == ret){
			break;
		}
		Chunk::popFrontPooled(m_chunks);
	}
	return ret;
}
void StreamBuffer::put(const void *data, std::size_t size){
	const char *read = (const char *)data;
	std::size_t copied = 0;

	AUTO(rit, m_chunks.rbegin());
	if((rit != m_chunks.rend()) && (rit->m_writePos < sizeof(rit->m_data))){
		const std::size_t toCopy = std::min(size - copied, sizeof(rit->m_data) - rit->m_writePos);
		std::memcpy(rit->m_data + rit->m_writePos, read, toCopy);
		rit->m_writePos += toCopy;
		read += toCopy;
		m_size += toCopy;
		copied += toCopy;
	}
	while(copied < size){
		AUTO_REF(back, Chunk::pushBackPooled(m_chunks));
		const std::size_t toCopy = std::min(size - copied, sizeof(back.m_data));
		std::memcpy(back.m_data, read, toCopy);
		back.m_writePos = toCopy;
		read += toCopy;
		m_size += toCopy;
		copied += toCopy;
	}
}
void StreamBuffer::put(const char *str){
	char ch;
	while((ch = *(str++)) != 0){
		put(ch);
	}
}
void StreamBuffer::put(const std::string &str){
	put(str.data(), str.size());
}

void StreamBuffer::swap(StreamBuffer &rhs) NOEXCEPT {
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
			const std::size_t avail = it->m_writePos - it->m_readPos;
			if(remaining < avail){
				AUTO_REF(back, Chunk::pushBackPooled(ret.m_chunks));
				std::memcpy(back.m_data, it->m_data + it->m_readPos, remaining);
				back.m_writePos = remaining;
				it->m_readPos += remaining;
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
void StreamBuffer::splice(StreamBuffer &src) NOEXCEPT {
	if(&src == this){
		return;
	}
	m_chunks.splice(m_chunks.end(), src.m_chunks);
	m_size += src.m_size;
	src.m_size = 0;
}

void StreamBuffer::dump(std::string &str) const {
	str.reserve(str.size() + size());
	for(AUTO(it, m_chunks.begin()); it != m_chunks.end(); ++it){
		str.append(it->m_data + it->m_readPos, it->m_data + it->m_writePos);
	}
}
void StreamBuffer::dump(std::ostream &os) const {
	for(AUTO(it, m_chunks.begin()); it != m_chunks.end(); ++it){
		os.write(it->m_data + it->m_readPos, it->m_writePos - it->m_readPos);
	}
}
void StreamBuffer::load(const std::string &str){
	clear();
	put(str.data(), str.size());
}
void StreamBuffer::load(std::istream &is){
	clear();
	for(;;){
		AUTO_REF(back, Chunk::pushBackPooled(m_chunks));
		back.m_writePos = is.readsome(back.m_data, sizeof(back.m_data));
		if(back.m_writePos == 0){
			break;
		}
	}
}

void StreamBuffer::dumpHex(std::ostream &os) const {
	static const char TABLE[] = "0123456789ABCDEF";
	char temp[2];
	for(AUTO(it, m_chunks.begin()); it != m_chunks.end(); ++it){
		for(std::size_t i = it->m_readPos; i < it->m_writePos; ++i){
			const unsigned byte = (unsigned char)it->m_data[i];
			temp[0] = TABLE[byte >> 4];
			temp[1] = TABLE[byte & 0x0F];
			os.write(temp, 2);
		}
	}
}
