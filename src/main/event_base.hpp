#ifndef POSEIDON_EVENT_BASE_HPP_
#define POSEIDON_EVENT_BASE_HPP_

#include <boost/enable_shared_from_this.hpp>

namespace Poseidon {

class EventBaseWithoutId
	: public boost::enable_shared_from_this<EventBaseWithoutId>
{
public:
	virtual ~EventBaseWithoutId();

public:
	virtual unsigned id() const = 0;

public:
	void raise();
};

template<unsigned EVENT_ID_T>
class EventBase : public EventBaseWithoutId {
public:
	enum {
		ID = EVENT_ID_T
	};

public:
	virtual unsigned id() const {
		return ID;
	}
};

}

#endif
