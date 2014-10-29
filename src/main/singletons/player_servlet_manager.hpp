#ifndef POSEIDON_SINGLETONS_PLAYER_SERVLET_MANAGER_HPP_
#define POSEIDON_SINGLETONS_PLAYER_SERVLET_MANAGER_HPP_

#include "../cxx_ver.hpp"
#include <boost/cstdint.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <boost/function.hpp>
#include "../stream_buffer.hpp"
#include "../player/protocol_base.hpp"

namespace Poseidon {

class PlayerServlet;
class PlayerSession;

typedef boost::function<
	void (boost::shared_ptr<PlayerSession> ps, StreamBuffer incoming)
	> PlayerServletCallback;

struct PlayerServletManager {
	static void start();
	static void stop();

	static std::size_t getMaxRequestLength();

	// 返回的 shared_ptr 是该响应器的唯一持有者。
	static boost::shared_ptr<PlayerServlet> registerServlet(
		unsigned port, boost::uint16_t protocolId, PlayerServletCallback callback);

	static boost::shared_ptr<const PlayerServletCallback> getServlet(
		unsigned port, boost::uint16_t protocolId);

private:
	PlayerServletManager();
};

}

#endif
