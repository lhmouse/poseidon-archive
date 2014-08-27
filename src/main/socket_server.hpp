#ifndef POSEIDON_SOCKET_SERVER_HPP_
#define POSEIDON_SOCKET_SERVER_HPP_

#include <string>
#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/ref.hpp>
#include <boost/utility.hpp>
#include <boost/type_traits/is_base_of.hpp>
#include "raii.hpp"
#include "virtual_shared_from_this.hpp"

namespace Poseidon {

class TcpSessionBase;

// 抽象工厂模式
class SocketServerBase : boost::noncopyable
	, public virtual VirtualSharedFromThis
{
private:
	std::string m_bindAddr;
	ScopedFile m_listen;

public:
	SocketServerBase(const std::string &bindAddr, unsigned bindPort);
	virtual ~SocketServerBase();

protected:
	// 工厂函数。
	// 如果该成员函数返回空指针，连接会被立即挂断。
	virtual boost::shared_ptr<TcpSessionBase> onClientConnect(ScopedFile &client) const = 0;

public:
	boost::shared_ptr<TcpSessionBase> tryAccept() const;
};

// onClientConnect() 返回值要求 DerivedTcpSessionBase 必须是 TcpSessionBase 的派生类。
template<class DerivedTcpSessionBase,
	typename boost::enable_if_c<
		boost::is_base_of<TcpSessionBase, DerivedTcpSessionBase>::value,
		int>::type = 0
	>
class SocketServer : public SocketServerBase {
public:
	SocketServer(const std::string &bindAddr, unsigned bindPort)
		: SocketServerBase(bindAddr, bindPort)
	{
	}

protected:
	virtual boost::shared_ptr<TcpSessionBase> onClientConnect(ScopedFile &client) const {
		return boost::make_shared<DerivedTcpSessionBase>(boost::ref(client));
	}
};

}

#endif
