#ifndef POSEIDON_SINGLETONS_PLAYER_SERVLET_MANAGER_HPP_
#define POSEIDON_SINGLETONS_PLAYER_SERVLET_MANAGER_HPP_

#include "../../cxx_ver.hpp"
#include <boost/cstdint.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include "../stream_buffer.hpp"

#ifdef POSEIDON_CXX11
#   include <functional>
#else
#   include <tr1/functional>
#endif

namespace Poseidon {

class PlayerServlet;
class PlayerSession;

typedef TR1::function<
	void (boost::shared_ptr<PlayerSession> ps, StreamBuffer incoming)
	> PlayerServletCallback;

struct PlayerServletManager {
	static void start();
	static void stop();

	static std::size_t getMaxRequestLength();

	// 返回的 shared_ptr 是该响应器的唯一持有者。
	// callback 禁止 move，否则可能出现主模块中引用子模块内存的情况。
	static boost::shared_ptr<PlayerServlet> registerServlet(boost::uint16_t protocolId,
		const boost::weak_ptr<const void> &dependency, const PlayerServletCallback &callback);

	static boost::shared_ptr<const PlayerServletCallback> getServlet(
		boost::shared_ptr<const void> &lockedDep, boost::uint16_t protocolId);

private:
	PlayerServletManager();
};

}

#endif
