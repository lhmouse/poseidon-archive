#ifndef POSEIDON_SINGLETONS_PLAYER_SERVLET_MANAGER_HPP_
#define POSEIDON_SINGLETONS_PLAYER_SERVLET_MANAGER_HPP_

#include <string>
#include <boost/cstdint.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <boost/function.hpp>
#include "../stream_buffer.hpp"

namespace Poseidon {

class PlayerServlet;

typedef boost::function<
	void (const boost::shared_ptr<class PlayerSession> &ps, StreamBuffer incoming)
	> PlayerServletCallback;

struct PlayerServletManager {
	// 返回的 shared_ptr 是该响应器的唯一持有者。
	// callback 禁止 move，否则可能出现主模块中引用子模块内存的情况。
	static boost::shared_ptr<const PlayerServlet>
		registerServlet(boost::uint16_t protocolId,
			const boost::weak_ptr<void> &dependency, const PlayerServletCallback &callback);

	static boost::shared_ptr<const PlayerServletCallback>
		getServlet(boost::shared_ptr<void> &lockedDep, boost::uint16_t protocolId);

private:
	PlayerServletManager();
};

}

#endif
