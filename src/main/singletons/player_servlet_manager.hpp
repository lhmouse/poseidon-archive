// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SINGLETONS_PLAYER_SERVLET_MANAGER_HPP_
#define POSEIDON_SINGLETONS_PLAYER_SERVLET_MANAGER_HPP_

#include "../cxx_ver.hpp"
#include <boost/cstdint.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <boost/function.hpp>
#include <boost/bind.hpp>
#include <boost/type_traits/is_base_of.hpp>
#include <boost/utility/enable_if.hpp>
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
		std::size_t category, boost::uint16_t protocolId, PlayerServletCallback callback);

	// void (boost::shared_ptr<PlayerSession> ps, ProtocolT request)
	template<typename ProtocolT, typename CallbackT>
	static
		typename boost::enable_if_c<boost::is_base_of<ProtocolBase, ProtocolT>::value,
			boost::shared_ptr<PlayerServlet> >::type
		registerServlet(std::size_t category,
#ifdef POSEIDON_CXX11
			CallbackT &&
#else
			const CallbackT &
#endif
			callback)
	{
		struct Helper {
			static void checkAndForward(
				boost::shared_ptr<PlayerSession> ps, StreamBuffer incoming, const CallbackT &callback)
			{
				return callback(STD_MOVE(ps), ProtocolT(incoming));
			}
		};
		return registerServlet(category, ProtocolT::ID, boost::bind(&Helper::checkAndForward, _1, _2,
#ifdef POSEIDON_CXX11
			std::forward<CallbackT>(callback)
#else
			callback
#endif
			));
	}

	static boost::shared_ptr<const PlayerServletCallback> getServlet(
		std::size_t category, boost::uint16_t protocolId);

private:
	PlayerServletManager();
};

}

#endif
