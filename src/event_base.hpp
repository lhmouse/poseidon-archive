// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_EVENT_BASE_HPP_
#define POSEIDON_EVENT_BASE_HPP_

#include <boost/shared_ptr.hpp>

namespace Poseidon {

class Event_base {
public:
	virtual ~Event_base();
};

extern void sync_raise_event(const boost::shared_ptr<Event_base> &event);
extern void async_raise_event(const boost::shared_ptr<Event_base> &event, const boost::shared_ptr<const bool> &withdrawn = boost::shared_ptr<const bool>());

}

#endif
