// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_EVENT_BASE_HPP_
#define POSEIDON_EVENT_BASE_HPP_

#include <boost/shared_ptr.hpp>

namespace Poseidon {

class EventBaseWithoutId {
public:
	virtual ~EventBaseWithoutId();

public:
	virtual unsigned get_id() const = 0;
};

template<unsigned EVENT_ID_T>
class EventBase : public EventBaseWithoutId {
public:
	enum {
		ID = EVENT_ID_T
	};

public:
	unsigned get_id() const OVERRIDE {
		return ID;
	}
};

extern void sync_raise_event(const boost::shared_ptr<EventBaseWithoutId> &event);
extern void async_raise_event(const boost::shared_ptr<EventBaseWithoutId> &event,
	const boost::shared_ptr<const bool> &withdrawn = boost::shared_ptr<const bool>());

}

#endif
