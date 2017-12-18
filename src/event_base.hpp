// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_EVENT_BASE_HPP_
#define POSEIDON_EVENT_BASE_HPP_

#include <boost/shared_ptr.hpp>

namespace Poseidon {

class EventBase {
public:
	virtual ~EventBase();
};

extern void sync_raise_event(const boost::shared_ptr<EventBase> &event);
extern void async_raise_event(const boost::shared_ptr<EventBase> &event, const boost::shared_ptr<const bool> &withdrawn = boost::shared_ptr<const bool>());

}

#endif
