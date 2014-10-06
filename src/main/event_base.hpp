#ifndef POSEIDON_EVENT_BASE_HPP_
#define POSEIDON_EVENT_BASE_HPP_

#include <boost/enable_shared_from_this.hpp>

namespace Poseidon {

class EventBase
	: public boost::enable_shared_from_this<EventBase>
{
public:
	virtual ~EventBase();

public:
	virtual unsigned id() const = 0;

public:
	void raise();
};

}

#endif
