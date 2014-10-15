#ifndef POSEIDON_TCP_SESSION_IMPL_
#	error Please do not #include "tcp_session_impl.hpp".
#endif

#ifndef POSEIDON_TCP_SESSION_IMPL_HPP_
#define POSEIDON_TCP_SESSION_IMPL_HPP_

namespace Poseidon {

struct TcpSessionImpl {
	static int doGetFd(const TcpSessionBase &session){
		return session.m_socket.get();
	}
	static long doRead(TcpSessionBase &session, void *buffer, unsigned long size){
		return session.doRead(buffer, size);
	}
	static long doWrite(TcpSessionBase &session, boost::mutex::scoped_lock &lock,
		void *hint, unsigned long hintSize)
	{
		return session.doWrite(lock, hint, hintSize);
	}
};

}

#endif
