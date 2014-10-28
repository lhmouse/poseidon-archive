#ifndef POSEIDON_SINGLETONS_PLAYER_SERVLET_MANAGER_HPP_
#define POSEIDON_SINGLETONS_PLAYER_SERVLET_MANAGER_HPP_

#include "../cxx_ver.hpp"
#include <boost/cstdint.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_base_of.hpp>
#include "../stream_buffer.hpp"
#include "../player/protocol_base.hpp"

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
	static boost::shared_ptr<PlayerServlet> registerServlet(unsigned port, boost::uint16_t protocolId,
		const boost::weak_ptr<const void> &dependency, const PlayerServletCallback &callback);

	template<typename ProtocolT>
	static boost::shared_ptr<PlayerServlet> registerServlet(unsigned port, boost::uint16_t protocolId,
		const boost::weak_ptr<const void> &dependency,
		const TR1::function<void (boost::shared_ptr<PlayerSession>, ProtocolT)> &callback)
	{
		BOOST_STATIC_ASSERT((boost::is_base_of<ProtocolBase, ProtocolT>::value));

		struct Helper {
			static void checkedForward(boost::shared_ptr<PlayerSession> ps, StreamBuffer incoming,
				const TR1::function<void (boost::shared_ptr<PlayerSession>, ProtocolT)> &derivedCallback)
			{
				derivedCallback(STD_MOVE(ps), ProtocolT(incoming));
			}
		};
		return registerServlet(port, protocolId, dependency,
			TR1::bind(&Helper::checkedForward, TR1::placeholders::_1, TR1::placeholders::_2, callback));
	}

	static boost::shared_ptr<const PlayerServletCallback> getServlet(unsigned port,
		boost::shared_ptr<const void> &lockedDep, boost::uint16_t protocolId);

private:
	PlayerServletManager();
};

}

#endif
