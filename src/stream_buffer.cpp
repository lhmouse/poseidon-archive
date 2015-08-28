// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "stream_buffer.hpp"
#include "atomic.hpp"

namespace Poseidon {

struct StreamBuffer::Chunk FINAL {
	static volatile bool s_poolLock;
	static Chunk *s_poolHead;

	static void *operator new(std::size_t bytes){
		assert(bytes == sizeof(Chunk));

		while(atomicExchange(s_poolLock, true, ATOMIC_ACQ_REL) == true){
			atomicPause();
		}
		const AUTO(head, s_poolHead);
		if(!head){
			atomicStore(s_poolLock, false, ATOMIC_RELEASE);
			return ::operator new(bytes);
		}
		s_poolHead = head->prev;
		atomicStore(s_poolLock, false, ATOMIC_RELEASE);
		return head;
	}
	static void operator delete(void *p) NOEXCEPT {
		if(!p){
			return;
		}
		const AUTO(head, static_cast<Chunk *>(p));

		while(atomicExchange(s_poolLock, true, ATOMIC_ACQ_REL) == true){
			atomicPause();
		}
		head->prev = s_poolHead;
		s_poolHead = head;
		atomicStore(s_poolLock, false, ATOMIC_RELEASE);
	}

	__attribute__((__destructor__))
	static void poolDestructor() NOEXCEPT {
		while(s_poolHead){
			const AUTO(head, s_poolHead);
			s_poolHead = head->prev;
			::operator delete(head);
		}
	}

	Chunk *prev;
	Chunk *next;
	unsigned begin;
	unsigned end;
	unsigned char data[0x100];
};

volatile bool StreamBuffer::Chunk::s_poolLock = false;
StreamBuffer::Chunk *StreamBuffer::Chunk::s_poolHead = NULLPTR;

unsigned char *StreamBuffer::ChunkEnumerator::begin() const NOEXCEPT {
	assert(m_chunk);

	return m_chunk->data + m_chunk->begin;
}
unsigned char *StreamBuffer::ChunkEnumerator::end() const NOEXCEPT {
	assert(m_chunk);

	return m_chunk->data + m_chunk->end;
}

StreamBuffer::ChunkEnumerator &StreamBuffer::ChunkEnumerator::operator++() NOEXCEPT {
	assert(m_chunk);

	m_chunk = m_chunk->next;
	return *this;
}

const unsigned char *StreamBuffer::ConstChunkEnumerator::begin() const NOEXCEPT {
	assert(m_chunk);

	return m_chunk->data + m_chunk->begin;
}
const unsigned char *StreamBuffer::ConstChunkEnumerator::end() const NOEXCEPT {
	assert(m_chunk);

	return m_chunk->data + m_chunk->end;
}

StreamBuffer::ConstChunkEnumerator &StreamBuffer::ConstChunkEnumerator::operator++() NOEXCEPT {
	assert(m_chunk);

	m_chunk = m_chunk->next;
	return *this;
}

// 构造函数和析构函数。
StreamBuffer::StreamBuffer(const void *data, std::size_t bytes)
	: m_first(NULLPTR), m_last(NULLPTR), m_size(0)
{
	put(data, bytes);
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
StreamBuffer::StreamBuffer(const StreamBuffer &rhs)
	: m_first(NULLPTR), m_last(NULLPTR), m_size(0)
{
	for(AUTO(ce, rhs.getChunkEnumerator()); ce; ++ce){
		put(ce.data(), ce.size());
	}
}
StreamBuffer &StreamBuffer::operator=(const StreamBuffer &rhs){
	StreamBuffer(rhs).swap(*this);
	return *this;
}
#ifdef POSEIDON_CXX11
StreamBuffer::StreamBuffer(StreamBuffer &&rhs) NOEXCEPT
	: StreamBuffer()
{
	swap(rhs);
}
StreamBuffer &StreamBuffer::operator=(StreamBuffer &&rhs) NOEXCEPT {
	StreamBuffer(std::move(rhs)).swap(*this);
	return *this;
}
#endif
StreamBuffer::~StreamBuffer(){
	clear();
}

// 其他非静态成员函数。
int StreamBuffer::front() const NOEXCEPT {
	if(m_size == 0){
		return -1;
	}

	int ret = -1;
	AUTO(chunk, m_first);
	do {
		if(chunk->end != chunk->begin){
			ret = chunk->data[chunk->begin];
		}
		chunk = chunk->next;
	} while(ret < 0);
	return ret;
}
int StreamBuffer::back() const NOEXCEPT {
	if(m_size == 0){
		return -1;
	}

	int ret = -1;
	AUTO(chunk, m_last);
	do {
		if(chunk->end != chunk->begin){
			ret = chunk->data[chunk->end - 1];
		}
		chunk = chunk->prev;
	} while(ret < 0);
	return ret;
}

void StreamBuffer::clear() NOEXCEPT {
	while(m_first){
		const AUTO(chunk, m_first);
		m_first = chunk->next;
		delete chunk;
	}
	m_last = NULLPTR;
	m_size = 0;
}

int StreamBuffer::get() NOEXCEPT {
	if(m_size == 0){
		return -1;
	}

	int ret = -1;
	AUTO(chunk, m_first);
	do {
		if(chunk->end != chunk->begin){
			ret = chunk->data[chunk->begin];
			++(chunk->begin);
		}
		if(chunk->begin == chunk->end){
			chunk = chunk->next;
			delete m_first;
			m_first = chunk;

			if(chunk){
				chunk->prev = NULLPTR;
			} else {
				m_last = NULLPTR;
			}
		}
	} while(ret < 0);
	m_size -= 1;
	return ret;
}
void StreamBuffer::put(unsigned char by){
	std::size_t lastAvail = 0;
	if(m_last){
		lastAvail = sizeof(m_last->data) - m_last->end;
	}
	Chunk *lastChunk = NULLPTR;
	if(lastAvail != 0){
		lastChunk = m_last;
	} else {
		AUTO(chunk, new Chunk);
		chunk->next = NULLPTR;
		// chunk->prev = NULLPTR;
		chunk->begin = 0;
		chunk->end = 0;

		if(m_last){
			m_last->next = chunk;
		} else {
			m_first = chunk;
		}
		chunk->prev = m_last;
		m_last = chunk;

		if(!lastChunk){
			lastChunk = chunk;
		}
	}

	AUTO(chunk, lastChunk);
	chunk->data[chunk->end] = by;
	++(chunk->end);
	++m_size;
}
int StreamBuffer::unput() NOEXCEPT {
	if(m_size == 0){
		return -1;
	}

	int ret = -1;
	AUTO(chunk, m_last);
	do {
		if(chunk->end != chunk->begin){
			--(chunk->end);
			ret = chunk->data[chunk->end];
		}
		if(chunk->begin == chunk->end){
			chunk = chunk->prev;
			delete m_last;
			m_last = chunk;

			if(chunk){
				chunk->next = NULLPTR;
			} else {
				m_first = NULLPTR;
			}
		}
	} while(ret < 0);
	m_size -= 1;
	return ret;
}
void StreamBuffer::unget(unsigned char by){
	std::size_t firstAvail = 0;
	if(m_first){
		firstAvail = m_first->begin;
	}
	Chunk *firstChunk = NULLPTR;
	if(firstAvail != 0){
		firstChunk = m_first;
	} else {
		AUTO(chunk, new Chunk);
		// chunk->next = NULLPTR;
		chunk->prev = NULLPTR;
		chunk->begin = sizeof(chunk->data);
		chunk->end = sizeof(chunk->data);

		if(m_first){
			m_first->prev = chunk;
		} else {
			m_last = chunk;
		}
		chunk->next = m_first;
		m_first = chunk;

		if(!firstChunk){
			firstChunk = chunk;
		}
	}

	AUTO(chunk, firstChunk);
	--(chunk->begin);
	chunk->data[chunk->begin] = by;
	++m_size;
}

std::size_t StreamBuffer::peek(void *data, std::size_t bytes) const NOEXCEPT {
	const AUTO(bytesToCopy, std::min(bytes, m_size));
	if(bytesToCopy == 0){
		return 0;
	}

	std::size_t bytesCopied = 0;
	AUTO(chunk, m_first);
	do {
		const AUTO(write, static_cast<unsigned char *>(data) + bytesCopied);
		const AUTO(bytesToCopyThisTime, std::min<std::size_t>(bytesToCopy - bytesCopied, chunk->end - chunk->begin));
		std::memcpy(write, chunk->data + chunk->begin, bytesToCopyThisTime);
		bytesCopied += bytesToCopyThisTime;
		chunk = chunk->next;
	} while(bytesCopied < bytesToCopy);
	return bytesCopied;
}
std::size_t StreamBuffer::get(void *data, std::size_t bytes) NOEXCEPT {
	const AUTO(bytesToCopy, std::min(bytes, m_size));
	if(bytesToCopy == 0){
		return 0;
	}

	std::size_t bytesCopied = 0;
	AUTO(chunk, m_first);
	do {
		const AUTO(write, static_cast<unsigned char *>(data) + bytesCopied);
		const AUTO(bytesToCopyThisTime, std::min<std::size_t>(bytesToCopy - bytesCopied, chunk->end - chunk->begin));
		std::memcpy(write, chunk->data + chunk->begin, bytesToCopyThisTime);
		bytesCopied += bytesToCopyThisTime;
		chunk->begin += bytesToCopyThisTime;
		if(chunk->begin == chunk->end){
			chunk = chunk->next;
			delete m_first;
			m_first = chunk;

			if(chunk){
				chunk->prev = NULLPTR;
			} else {
				m_last = NULLPTR;
			}
		}
	} while(bytesCopied < bytesToCopy);
	m_size -= bytesToCopy;
	return bytesCopied;
}
std::size_t StreamBuffer::discard(std::size_t bytes) NOEXCEPT {
	const AUTO(bytesToCopy, std::min(bytes, m_size));
	if(bytesToCopy == 0){
		return 0;
	}

	std::size_t bytesCopied = 0;
	AUTO(chunk, m_first);
	do {
		const AUTO(bytesToCopyThisTime, std::min<std::size_t>(bytesToCopy - bytesCopied, chunk->end - chunk->begin));
		bytesCopied += bytesToCopyThisTime;
		chunk->begin += bytesToCopyThisTime;
		if(chunk->begin == chunk->end){
			chunk = chunk->next;
			delete m_first;
			m_first = chunk;

			if(chunk){
				chunk->prev = NULLPTR;
			} else {
				m_last = NULLPTR;
			}
		}
	} while(bytesCopied < bytesToCopy);
	m_size -= bytesToCopy;
	return bytesCopied;
}
void StreamBuffer::put(const void *data, std::size_t bytes){
	const AUTO(bytesToCopy, bytes);
	if(bytesToCopy == 0){
		return;
	}

	std::size_t lastAvail = 0;
	if(m_last){
		lastAvail = sizeof(m_last->data) - m_last->end;
	}
	Chunk *lastChunk = NULLPTR;
	if(lastAvail != 0){
		lastChunk = m_last;
	}
	if(bytesToCopy > lastAvail){
		const AUTO(newChunks, (bytesToCopy - lastAvail - 1) / sizeof(lastChunk->data) + 1);
		assert(newChunks != 0);

		AUTO(chunk, new Chunk);
		chunk->next = NULLPTR;
		chunk->prev = NULLPTR;
		chunk->begin = 0;
		chunk->end = 0;

		AUTO(spliceFirst, chunk), spliceLast = chunk;
		try {
			for(std::size_t i = 1; i < newChunks; ++i){
				chunk = new Chunk;
				chunk->next = NULLPTR;
				chunk->prev = spliceLast;
				chunk->begin = 0;
				chunk->end = 0;

				spliceLast->next = chunk;
				spliceLast = chunk;
			}
		} catch(...){
			do {
				chunk = spliceFirst;
				spliceFirst = chunk->next;
				delete chunk;
			} while(spliceFirst);

			throw;
		}
		if(m_last){
			m_last->next = spliceFirst;
		} else {
			m_first = spliceFirst;
		}
		spliceFirst->prev = m_last;
		m_last = spliceLast;

		if(!lastChunk){
			lastChunk = spliceFirst;
		}
	}

	std::size_t bytesCopied = 0;
	AUTO(chunk, lastChunk);
	do {
		const AUTO(read, static_cast<const unsigned char *>(data) + bytesCopied);
		const AUTO(bytesToCopyThisTime, std::min<std::size_t>(bytesToCopy - bytesCopied, sizeof(chunk->data) - chunk->end));
		std::memcpy(chunk->data + chunk->end, read, bytesToCopyThisTime);
		chunk->end += bytesToCopyThisTime;
		bytesCopied += bytesToCopyThisTime;
		chunk = chunk->next;
	} while(bytesCopied < bytesToCopy);
	m_size += bytesToCopy;
}
void StreamBuffer::put(const char *str){
	put(str, std::strlen(str));
}
void StreamBuffer::put(const std::string &str){
	put(str.data(), str.size());
}

StreamBuffer StreamBuffer::cutOff(std::size_t bytes){
	StreamBuffer ret;

	const AUTO(bytesToCopy, std::min(bytes, m_size));
	if(bytesToCopy == 0){
		return ret;
	}

	if(m_size <= bytesToCopy){
		ret.swap(*this);
		return ret;
	}

	std::size_t bytesCopied = 0;
	AUTO(cutEnd, m_first);
	for(;;){
		const AUTO(bytesRemaining, bytesToCopy - bytesCopied);
		const AUTO(bytesAvail, cutEnd->end - cutEnd->begin);
		if(bytesRemaining <= bytesAvail){
			if(bytesRemaining == bytesAvail){
				cutEnd = cutEnd->next;
			} else {
				const AUTO(chunk, new Chunk);
				chunk->next = cutEnd;
				chunk->prev = cutEnd->prev;
				chunk->begin = 0;
				chunk->end = bytesRemaining;

				std::memcpy(chunk->data, cutEnd->data + cutEnd->begin, bytesRemaining);
				cutEnd->begin += bytesRemaining;

				if(cutEnd->prev){
					cutEnd->prev->next = chunk;
				} else {
					m_first = chunk;
				}
				cutEnd->prev = chunk;
			}
			break;
		}
		bytesCopied += bytesAvail;
		cutEnd = cutEnd->next;
	}

	const AUTO(cutFirst, m_first);
	const AUTO(cutLast, cutEnd->prev);
	cutLast->next = NULLPTR;
	cutEnd->prev = NULLPTR;

	m_first = cutEnd;
	m_size -= bytesToCopy;

	ret.m_first = cutFirst;
	ret.m_last = cutLast;
	ret.m_size = bytesToCopy;
	return ret;
}
void StreamBuffer::splice(StreamBuffer &rhs) NOEXCEPT {
	if(&rhs == this){
		return;
	}
	if(!rhs.m_first){
		return;
	}

	if(m_last){
		m_last->next = rhs.m_first;
	} else {
		m_first = rhs.m_first;
	}
	rhs.m_first->prev = m_last;
	m_last = rhs.m_last;
	m_size += rhs.m_size;

	rhs.m_first = NULLPTR;
	rhs.m_last = NULLPTR;
	rhs.m_size = 0;
}

}
