#ifndef POSEIDON_SINGLETONS_WEBSOCKET_SERVLET_MANAGER_HPP_
#define POSEIDON_SINGLETONS_WEBSOCKET_SERVLET_MANAGER_HPP_

#include "../../cxx_ver.hpp"
#include <string>
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include "../stream_buffer.hpp"
#include "../http/websocket/opcode.hpp"

#ifdef POSEIDON_CXX11
#   include <functional>
#else
#   include <tr1/functional>
#endif

namespace Poseidon {

class WebSocketServlet;
class WebSocketSession;

typedef TR1::function<
	void (boost::shared_ptr<WebSocketSession> wss, WebSocketOpCode opcode, StreamBuffer incoming)
	> WebSocketServletCallback;

struct WebSocketServletManager {
	static void start();
	static void stop();

	// 返回的 shared_ptr 是该响应器的唯一持有者。
	// callback 禁止 move，否则可能出现主模块中引用子模块内存的情况。
	static boost::shared_ptr<const WebSocketServlet>
		registerServlet(const std::string &uri,
			const boost::weak_ptr<const void> &dependency, const WebSocketServletCallback &callback);

	static boost::shared_ptr<const WebSocketServletCallback>
		getServlet(boost::shared_ptr<const void> &lockedDep, const std::string &uri);

private:
	WebSocketServletManager();
};

}

#endif
