#ifndef POSEIDON_EPOLL_DAEMON_HPP_
#define POSEIDON_EPOLL_DAEMON_HPP_

#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>
#include "../raii.hpp"

namespace Poseidon {

class TcpPeer;

struct EpollDaemon {
	static void start();
	static void stop();

	// 加入 epoll，一旦有数据则调用其 onDataAvail() 成员函数，
	// 对于写事件采用延迟写入与写入合并策略。
	// 一旦出错或被挂断则从 epoll 中移除。
	static void addPeer(const boost::shared_ptr<TcpPeer> &peer);
	// 如果回调返回 false 那么这个回调就会被删掉。
	static void registerIdleCallback(boost::function<bool ()> callback);

private:
	EpollDaemon();
};

}

#endif
