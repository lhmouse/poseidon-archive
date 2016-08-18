// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "oid.hpp"
#include <sys/types.h>
#include <unistd.h>
#include "../atomic.hpp"
#include "../endian.hpp"
#include "../exception.hpp"
#include "../time.hpp"
#include "../random.hpp"

namespace Poseidon {

namespace {
	const boost::uint32_t g_mid = rand32();
	const boost::uint32_t g_pid = static_cast<boost::uint16_t>(::getpid());

	volatile boost::uint32_t g_auto_inc = 0;
}

namespace MongoDb {
	Oid Oid::random(){
		const AUTO(utc_now, get_utc_time());
		const AUTO(auto_inc, atomic_add(g_auto_inc, 1, ATOMIC_RELAXED));

#define COPY_BE(field_, src_)	\
		do {	\
			boost::uint32_t temp_;	\
			store_be(temp_, src_);	\
			std::memcpy(&(field_), &temp_, sizeof(field_));	\
		} while(0)

		Oid ret;
		COPY_BE(ret.m_storage.uts, utc_now / 1000);
		COPY_BE(ret.m_storage.mid, g_mid);
		COPY_BE(ret.m_storage.pid, g_pid);
		COPY_BE(ret.m_storage.inc, auto_inc);
		return ret;
	}
}

}
