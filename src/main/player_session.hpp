#ifndef POSEIDON_PLAYER_SESSION_HPP_
#define POSEIDON_PLAYER_SESSION_HPP_

#include <boost/shared_ptr.hpp>
#include <boost/nopncopyable.hpp>

namespace Poseidon {

class PlayerSession : boost::noncopyable {
private:

public:
	explicit PlayerSession(ScopedFile &socket);

protected:
	void onReadAvail(const void *data, std::size_t size);
};

}

#endif
