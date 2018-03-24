// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "udp_session_base.hpp"
#include "singletons/epoll_daemon.hpp"
#include "log.hpp"
#include "system_exception.hpp"
#include "profiler.hpp"
#include "errno.hpp"
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/udp.h>
#include <net/if.h>

namespace Poseidon {

namespace {
	inline int ipproto_of(const UdpSessionBase *session){
		return session->is_using_ipv6() ? IPPROTO_IPV6 : IPPROTO_IP;
	}

	struct InterfaceListFreeer {
		CONSTEXPR struct ::if_nameindex *operator()() NOEXCEPT {
			return NULLPTR;
		}
		void operator()(struct ::if_nameindex *ptr) NOEXCEPT {
			::if_freenameindex(ptr);
		}
	};

	UniqueHandle<InterfaceListFreeer> get_all_interfaces(){
		PROFILE_ME;

		UniqueHandle<InterfaceListFreeer> list;
		DEBUG_THROW_UNLESS(list.reset(::if_nameindex()), SystemException);
		return list;
	}

	union Buffer_mreq {
		::ipv6_mreq v6;
		::ip_mreqn v4;
	};

	::socklen_t make_mreq_opt(Buffer_mreq *opt, const SockAddr &group, unsigned if_index){
		if(group.is_ipv6()){
			opt->v6.ipv6mr_multiaddr = static_cast<const ::sockaddr_in6 *>(group.data())->sin6_addr;
			opt->v6.ipv6mr_interface = if_index;
			return sizeof(opt->v6);
		} else {
			opt->v4.imr_multiaddr = static_cast<const ::sockaddr_in *>(group.data())->sin_addr;
			opt->v4.imr_address.s_addr = INADDR_ANY;
			opt->v4.imr_ifindex = static_cast<int>(if_index);
			return sizeof(opt->v4);
		}
	}
}

UdpSessionBase::UdpSessionBase(Move<UniqueFile> socket)
	: SocketBase(STD_MOVE(socket))
{
	//
}
UdpSessionBase::~UdpSessionBase(){
	//
}

bool UdpSessionBase::has_been_shutdown_read() const NOEXCEPT {
	return SocketBase::has_been_shutdown_read();
}
bool UdpSessionBase::has_been_shutdown_write() const NOEXCEPT {
	return SocketBase::has_been_shutdown_write();
}
bool UdpSessionBase::shutdown_read() NOEXCEPT {
	return SocketBase::shutdown_read();
}
bool UdpSessionBase::shutdown_write() NOEXCEPT {
	return SocketBase::shutdown_write();
}
void UdpSessionBase::force_shutdown() NOEXCEPT {
	SocketBase::force_shutdown();
}

int UdpSessionBase::poll_read_and_process(unsigned char *hint_buffer, std::size_t hint_capacity, bool /*readable*/){
	PROFILE_ME;

	for(unsigned i = 0; i < 256; ++i){
		SockAddr sock_addr;
		StreamBuffer data;
		try {
			::sockaddr_storage sa;
			::socklen_t sa_len = sizeof(sa);
			::ssize_t result = ::recvfrom(get_fd(), hint_buffer, hint_capacity, MSG_NOSIGNAL | MSG_DONTWAIT, static_cast< ::sockaddr *>(static_cast<void *>(&sa)), &sa_len);
			if(result < 0){
				return errno;
			}
			sock_addr = SockAddr(&sa, sa_len);
			data.put(hint_buffer, static_cast<std::size_t>(result));
			LOG_POSEIDON_TRACE("Read ", result, " byte(s) from ", IpPort(sock_addr));
		} catch(std::exception &e){
			LOG_POSEIDON_ERROR("std::exception thrown: what = ", e.what());
			return EINTR;
		}
		try {
			on_receive(sock_addr, STD_MOVE(data));
		} catch(std::exception &e){
			LOG_POSEIDON_ERROR("std::exception thrown: what = ", e.what());
			continue;
		} catch(...){
			LOG_POSEIDON_ERROR("Unknown exception thrown.");
			continue;
		}
	}
	return 0;
}
int UdpSessionBase::poll_write(Mutex::UniqueLock &/*write_lock*/, unsigned char *hint_buffer, std::size_t hint_capacity, bool /*writeable*/){
	PROFILE_ME;

	for(unsigned i = 0; i < 256; ++i){
		SockAddr sock_addr;
		StreamBuffer data;
		try {
			const Mutex::UniqueLock lock(m_send_mutex);
			if(m_send_queue.empty()){
				return EWOULDBLOCK;
			}
			sock_addr = m_send_queue.front().first;
			data.swap(m_send_queue.front().second);
			m_send_queue.pop_front();
		} catch(std::exception &e){
			LOG_POSEIDON_ERROR("std::exception thrown: what = ", e.what());
			return EINTR;
		}
		try {
			::sockaddr_storage sa;
			DEBUG_THROW_ASSERT(sock_addr.size() <= sizeof(sa));
			::socklen_t sa_len = static_cast<unsigned>(sock_addr.size());
			const std::size_t avail = data.peek(hint_buffer, hint_capacity);
			if(avail < data.size()){
				LOG_POSEIDON(Logger::special_major | Logger::level_debug, "UDP packet is too large: size = ", data.size());
		_too_large:
				on_message_too_large(sock_addr, STD_MOVE(data));
				continue;
			}
			::ssize_t result = ::sendto(get_fd(), hint_buffer, avail, MSG_NOSIGNAL | MSG_DONTWAIT, static_cast< ::sockaddr *>(static_cast<void *>(&sa)), sa_len);
			if(result < 0){
				if(errno == EMSGSIZE){
					LOG_POSEIDON(Logger::special_major | Logger::level_debug, "UDP packet is too large: size = ", data.size());
					goto _too_large;
				}
				continue;
			}
			LOG_POSEIDON_TRACE("Wrote ", result, " byte(s) to ", IpPort(sock_addr));
		} catch(std::exception &e){
			LOG_POSEIDON_ERROR("std::exception thrown: what = ", e.what());
			continue;
		}
	}
	return 0;
}

void UdpSessionBase::on_message_too_large(const SockAddr &/*sock_addr*/, StreamBuffer /*data*/){
	//
}

void UdpSessionBase::add_membership(const SockAddr &group){
	PROFILE_ME;
	DEBUG_THROW_UNLESS(is_using_ipv6() == group.is_ipv6(), Exception, sslit("Socket family does not match the group address provided"));

	const AUTO(interfaces, get_all_interfaces());
	Buffer_mreq opt;
	AUTO(ptr, interfaces.get());
	try {
		while(ptr->if_name){
			LOG_POSEIDON_DEBUG("Adding UDP socket to multicast group: group = ", IpPort(group), ", interface = ", ptr->if_name);
			const AUTO(optlen, make_mreq_opt(&opt, group, ptr->if_index));
			if(::setsockopt(get_fd(), ipproto_of(this), is_using_ipv6() ? IPV6_ADD_MEMBERSHIP : IP_ADD_MEMBERSHIP, &opt, optlen) != 0){
				const int err_code = errno;
				LOG_POSEIDON_ERROR("::setsockopt() failed, errno was ", err_code, " (", get_error_desc(err_code), ")");
				DEBUG_THROW(SystemException, err_code);
			}
			++ptr;
		}
	} catch(...){
		while(ptr != interfaces.get()){
			--ptr;
			LOG_POSEIDON_DEBUG("Removing UDP socket from multicast group: group = ", IpPort(group), ", interface = ", ptr->if_name);
			const AUTO(optlen, make_mreq_opt(&opt, group, ptr->if_index));
			if(::setsockopt(get_fd(), ipproto_of(this), is_using_ipv6() ? IPV6_DROP_MEMBERSHIP : IP_DROP_MEMBERSHIP, &opt, optlen) != 0){
				const int err_code = errno;
				LOG_POSEIDON_WARNING("::setsockopt() failed, errno was ", err_code, " (", get_error_desc(err_code), ")");
			}
		}
		throw;
	}
}
void UdpSessionBase::drop_membership(const SockAddr &group){
	PROFILE_ME;
	DEBUG_THROW_UNLESS(is_using_ipv6() == group.is_ipv6(), Exception, sslit("Socket family does not match the group address provided"));

	const AUTO(interfaces, get_all_interfaces());
	Buffer_mreq opt;
	AUTO(ptr, interfaces.get());
	while(ptr->if_name){
		LOG_POSEIDON_DEBUG("Removing UDP socket from multicast group: group = ", IpPort(group), ", interface = ", ptr->if_name);
		const AUTO(optlen, make_mreq_opt(&opt, group, ptr->if_index));
		if(::setsockopt(get_fd(), ipproto_of(this), is_using_ipv6() ? IPV6_DROP_MEMBERSHIP : IP_DROP_MEMBERSHIP, &opt, optlen) != 0){
			const int err_code = errno;
			LOG_POSEIDON_WARNING("::setsockopt() failed, errno was ", err_code, " (", get_error_desc(err_code), ")");
		}
		++ptr;
	}
}
void UdpSessionBase::set_multicast_loop(bool enabled){
	const int val = enabled;
	DEBUG_THROW_UNLESS(::setsockopt(get_fd(), ipproto_of(this), is_using_ipv6() ? IPV6_MULTICAST_LOOP : IP_MULTICAST_LOOP, &val, sizeof(val)) == 0, SystemException);
}
void UdpSessionBase::set_multicast_ttl(int ttl){
	const int val = ttl;
	DEBUG_THROW_UNLESS(::setsockopt(get_fd(), ipproto_of(this), is_using_ipv6() ? IPV6_MULTICAST_HOPS : IP_MULTICAST_TTL, &val, sizeof(val)) == 0, SystemException);
}

bool UdpSessionBase::send(const SockAddr &sock_addr, StreamBuffer buffer){
	PROFILE_ME;

	if(has_been_shutdown_write()){
		LOG_POSEIDON(Logger::special_major | Logger::level_debug, "UDP socket has been shut down for writing: local = ", get_local_info(), ", remote = ", IpPort(sock_addr));
		return false;
	}

	const Mutex::UniqueLock lock(m_send_mutex);
	m_send_queue.emplace_back(sock_addr, STD_MOVE(buffer));
	EpollDaemon::mark_socket_writeable(this);
	return true;
}

}
