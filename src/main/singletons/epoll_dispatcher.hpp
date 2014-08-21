#ifndef POSEIDON_EPOLL_DISPATCHER_HPP_
#define POSEIDON_EPOLL_DISPATCHER_HPP_

#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>
#include "../raii.hpp"

namespace Poseidon {

class TcpPeer;

struct EpollDispatcher {
	static void startDaemon();
	static void stopDaemon();

	// 加入 epoll，一旦有数据则调用其 onDataAvail() 成员函数，
	// 一旦出错或被挂断则从 epoll 中移除。
	static void registerTcpPeer(const boost::shared_ptr<TcpPeer> &readable);
	// 延迟写入与写入合并。
	static void pendWrite(const boost::shared_ptr<TcpPeer> &writeable);
	// 如果回调返回 false 那么这个回调就会被删掉。
	static void registerIdleCallback(boost::function<bool ()> callback);

private:
	EpollDispatcher();
};

}

#endif
