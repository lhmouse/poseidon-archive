#ifndef POSEIDON_TCP_SERVER_HPP_
#define POSEIDON_TCP_SERVER_HPP_

#include <string>
#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>
#include "raii.hpp"

namespace Poseidon {

// 抽象工厂模式
class TcpServerBase : boost::noncopyable {
private:
	std::string m_bindAddr;
	ScopedFile m_listen;

public:
	TcpServerBase(const std::string &bindAddr, unsigned bindPort);
	virtual ~TcpServerBase();

protected:
	// 工厂函数。
	// 如果该成员函数返回空指针，连接会被立即挂断。
	virtual boost::shared_ptr<class TcpSessionBase>
		onClientConnect(Move<ScopedFile> client) const = 0;

public:
	boost::shared_ptr<TcpSessionBase> tryAccept() const;
};

}

#endif
