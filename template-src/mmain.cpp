// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

// 这个文件被置于公有领域（public domain）。
/*
#include "../src/precompiled.hpp"
#include "../src/log.hpp"
#include "../src/exception.hpp"
#include "../src/profiler.hpp"
#include "../src/hash.hpp"
#include "../src/module_raii.hpp"
#include "../src/singletons/module_depository.hpp"
#include "../src/singletons/event_dispatcher.hpp"
#include "../src/singletons/timer_daemon.hpp"
#include "../src/singletons/profile_depository.hpp"
#include "../src/websocket/session.hpp"
#include "../src/http/utilities.hpp"
#include "../src/tcp_client_base.hpp"
#include "../src/cbpp/session.hpp"
#include "../src/http/session.hpp"
#include "../src/cbpp/message_base.hpp"
#include "../src/singletons/epoll_daemon.hpp"
#include "../src/mysql/object_base.hpp"
#include "../src/job_base.hpp"
#include "../src/uuid.hpp"
#include "../src/async_job.hpp"
using namespace Poseidon;

#define MYSQL_OBJECT_NAME	MySqlObj
#define MYSQL_OBJECT_FIELDS	\
	FIELD_SMALLINT(si)	\
	FIELD_STRING(str)	\
	FIELD_BIGINT(bi)	\
	FIELD_DATETIME(dt)
#include "../src/mysql/object_generator.hpp"

namespace {
	void loaded_proc(bool found){
		LOG_POSEIDON_FATAL("-- loaded! found = ", found);
	}

	void write(){
		AUTO(obj, boost::make_shared<MySqlObj>());
		obj->enable_auto_saving();
		obj->set_si(999);
		obj->set_str("meow");
		for(int i = 0; i < 10; ++i){
			obj->set_bi(i);
		}
		obj->async_load("SELECT * FROM `MySqlObj`", loaded_proc);
	}
}
MODULE_RAII {
	enqueue_async_job(write, 3000);
	return VAL_INIT;
}

namespace {

struct TestEvent1 : public EventBase<1> {
	int i;
	std::string s;
};

struct TestEvent2 : public EventBase<1> {
	double d;
};

void event1_proc(boost::shared_ptr<TestEvent1> event){
	PROFILE_ME;
	LOG_POSEIDON_FATAL("event1_proc: i = ", event->i, ", s = ", event->s);
}

void event2_proc(boost::shared_ptr<TestEvent2> event){
	PROFILE_ME;
	LOG_POSEIDON_FATAL("event2_proc: d = ", event->d);
}

void print_objs(std::vector<boost::shared_ptr<MySqlObjFectBase> > v){
	LOG_POSEIDON_FATAL("--------- v.size() = ", v.size());
	for(AUTO(it, v.begin()); it != v.end(); ++it){
		AUTO(p, static_cast<MySqlObjF *>(it->get()));
		LOG_POSEIDON_FATAL("-- si = ", p->get_si(), ", str = ", p->get_str(), ", bi = ", p->get_bi());
	}
}

void mysql_except_proc(){
	LOG_POSEIDON_FATAL("MySQL exception!");
}

void tick_proc(unsigned long long now, unsigned long long period){
	PROFILE_ME;
	LOG_POSEIDON_FATAL("Tick, now = ", now, ", period = ", period);

	MySqlObjF::batch_load("SELECT * FROM `MySqlObjF`", &print_objs, &mysql_except_proc);
}

void profile_proc(boost::shared_ptr<HttpSession> hs, HttpRequest){
	PROFILE_ME;

	OptionalMap headers;
	StreamBuffer contents;

	headers.set("Content-Type", "text/plain");
	contents.put("   Samples      Total time(us)  Exclusive time(us)    File:Line\n");
	const AUTO(profile, ProfileDepository::snapshot());
	for(AUTO(it, profile.begin()); it != profile.end(); ++it){
		char temp[128];
		unsigned len = (unsigned)std::sprintf(temp, "%10llu%20llu%20llu    ",
			it->samples, it->us_total, it->us_exclusive);
		contents.put(temp, len);
		contents.put(it->file);
		len = (unsigned)std::sprintf(temp, ":%lu\n", it->line);
		contents.put(temp, len);
	}
	hs->send(HTTP_OK, STD_MOVE(headers), STD_MOVE(contents));
}

void meow_proc(boost::shared_ptr<HttpSession> hs, HttpRequest){
	PROFILE_ME;

	AUTO(obj, boost::make_shared<MySqlObjF>());
	obj->set_si(123);
	obj->set_str("meow");
	obj->set_bi(456789);
	obj->async_save(true);

	OptionalMap headers;
	StreamBuffer contents;

	AUTO(event, boost::make_shared<TestEvent1>());
	event->i = 123;
	event->s = "meow";
	event->raise();

	headers.set("Content-Type", "text/html");
	contents.put("<h1>Meow!</h1>");
	hs->send(HTTP_OK, STD_MOVE(headers), STD_MOVE(contents));
}
void meow_meow_proc(boost::shared_ptr<HttpSession> hs, HttpRequest){
	PROFILE_ME;

	OptionalMap headers;
	StreamBuffer contents;

	AUTO(event, boost::make_shared<TestEvent1>());
	event->i = 123;
	event->s = "meow/meow";
	event->raise();

	headers.set("Content-Type", "text/html");
	contents.put("<h1>Meow! Meow!</h1>");
	hs->send(HTTP_OK, STD_MOVE(headers), STD_MOVE(contents));
}

boost::weak_ptr<std::vector<boost::shared_ptr<void> > > g_servlets;

void load_proc(boost::shared_ptr<HttpSession> hs, HttpRequest){
	PROFILE_ME;

	OptionalMap headers;
	StreamBuffer contents;

	AUTO(event, boost::make_shared<TestEvent2>());
	event->d = 123.45;
	event->raise();

	AUTO_REF(v, *boost::shared_ptr<std::vector<boost::shared_ptr<void> > >(g_servlets));
	if(!v.empty()){
		contents.put("Already loaded");
	} else {
		v.push_back(HttpServletDepository::create(1, SSLIT("/meow/"), &meow_proc));
		v.push_back(HttpServletDepository::create(1, SSLIT("/meow/meow/"), &meow_meow_proc));
		v.push_back(TimerDaemon::register_timer(5000, 10000, &tick_proc));
		contents.put("OK");
	}
	hs->send(HTTP_OK, STD_MOVE(headers), STD_MOVE(contents));
}
void unload_proc(boost::shared_ptr<HttpSession> hs, HttpRequest){
	PROFILE_ME;

	OptionalMap headers;
	StreamBuffer contents;

	AUTO(event, boost::make_shared<TestEvent2>());
	event->d = 67.89;
	event->raise();

	AUTO_REF(v, *boost::shared_ptr<std::vector<boost::shared_ptr<void> > >(g_servlets));
	if(v.empty()){
		contents.put("Already unloaded");
	} else {
		v.clear();
		contents.put("OK");
	}
	hs->send(HTTP_OK, STD_MOVE(headers), STD_MOVE(contents));
}

void web_socket_proc(boost::shared_ptr<WebSocketSession> wss,
	WebSocketOpCode opcode, StreamBuffer incoming)
{
	PROFILE_ME;
	LOG_POSEIDON_FATAL("Received packet: opcode = ", opcode, ", payload = ", StreamBufferHexDumper(incoming));

	std::string s;
	incoming.dump(s);

	char crc32_str[16];
	std::sprintf(crc32_str, "%08lx", (unsigned long)crc32_sum(s.data(), s.size()));

	unsigned char md5[16];
	md5_sum(md5, s.data(), s.size());
	std::string md5_str = hex_encode(md5, sizeof(md5));

	unsigned char sha1[20];
	sha1_sum(sha1, s.data(), s.size());
	std::string sha1_str = hex_encode(sha1, sizeof(sha1));

	StreamBuffer out;
	out.put("CRC32: ");
	out.put(crc32_str);
	out.put("\nMD5: ");
	out.put(md5_str.data(), md5_str.size());
	out.put("\nSHA1: ");
	out.put(sha1_str.data(), sha1_str.size());
	out.put('\n');
	wss->send(STD_MOVE(out), false);
}

class TestClient : public TcpClientBase {
public:
	static boost::shared_ptr<TestClient> create(){
		AUTO(ret, boost::make_shared<TestClient>());
		ret->go_resident();
		return ret;
	}

public:
	TestClient()
		: TcpClientBase(IpPort(SSLIT("192.30.252.128"), 443), true)
	{
	}

private:
	void on_read_avail(const void *data, std::size_t size){
		AUTO(read, (const char *)data);
		for(std::size_t i = 0; i < size; ++i){
			std::putchar(read[i]);
		}
	}
};

}

#define MESSAGE_NAME		TestInt
#define MESSAGE_ID			100
#define MESSAGE_FIELDS		FIELD_VINT(i)
#include "../src/cbpp/message_generator.hpp"

#define MESSAGE_NAME		TestUInt
#define MESSAGE_ID			101
#define MESSAGE_FIELDS 	FIELD_VUINT(u)
#include "../src/cbpp/message_generator.hpp"

#define MESSAGE_NAME		TestString
#define MESSAGE_ID			102
#define MESSAGE_FIELDS		FIELD_STRING(s)
#include "../src/cbpp/message_generator.hpp"

#define MESSAGE_NAME		TestIntArray
#define MESSAGE_ID			103
#define MESSAGE_FIELDS		FIELD_ARRAY(a, FIELD_VINT(i))
#include "../src/cbpp/message_generator.hpp"

#define MESSAGE_NAME		TestUIntArray
#define MESSAGE_ID			104
#define MESSAGE_FIELDS		FIELD_ARRAY(a, FIELD_VUINT(u))
#include "../src/cbpp/message_generator.hpp"

#define MESSAGE_NAME		TestStringArray
#define MESSAGE_ID			105
#define MESSAGE_FIELDS		FIELD_ARRAY(a, FIELD_STRING(s))
#include "../src/cbpp/message_generator.hpp"

#define MESSAGE_NAME	   	TestMessage
#define MESSAGE_ID			106
#define MESSAGE_FIELDS \
	FIELD_VINT(i)   \
	FIELD_VUINT(j)  \
	FIELD_ARRAY(a,  \
		FIELD_BYTES(b, 3)	\
		FIELD_STRING(s) \
		FIELD_VUINT(k)  \
	)
#include "../src/cbpp/message_generator.hpp"

namespace {

void TestIntProc(boost::shared_ptr<CbppSession> ps, StreamBuffer incoming){
	TestInt req(incoming);
	LOG_POSEIDON_WARNING("sint = ", req.i);
	req.i /= 10;
	ps->send(1000, req);
}
void TestUIntProc(boost::shared_ptr<CbppSession> ps, StreamBuffer incoming){
	TestUInt req(incoming);
	LOG_POSEIDON_WARNING("int = ", req.u);
	req.u /= 10;
	ps->send(1001, req);
}
void TestStringProc(boost::shared_ptr<CbppSession> ps, StreamBuffer incoming){
	TestString req(incoming);
	LOG_POSEIDON_WARNING("string = ", req.s);
	req.s += "_0123456789";
	ps->send(1002, req);
}

void TestIntArrayProc(boost::shared_ptr<CbppSession> ps, StreamBuffer incoming){
	TestIntArray req(incoming);
	LOG_POSEIDON_WARNING("sint array: size = ", req.a.size());
	for(std::size_t i = 0; i < req.a.size(); ++i){
		LOG_POSEIDON_WARNING("  ", i, " = ", req.a.at(i).i);
		req.a.at(i).i /= 10;
	}
	ps->send(1003, req);
}
void TestUIntArrayProc(boost::shared_ptr<CbppSession> ps, StreamBuffer incoming){
	TestUIntArray req(incoming);
	LOG_POSEIDON_WARNING("sint array: size = ", req.a.size());
	for(std::size_t i = 0; i < req.a.size(); ++i){
		LOG_POSEIDON_WARNING("  ", i, " = ", req.a.at(i).u);
		req.a.at(i).u /= 10;
	}
	ps->send(1004, req);
}
void TestStringArrayProc(boost::shared_ptr<CbppSession> ps, StreamBuffer incoming){
	TestStringArray req(incoming);
	LOG_POSEIDON_WARNING("sint array: size = ", req.a.size());
	for(std::size_t i = 0; i < req.a.size(); ++i){
		LOG_POSEIDON_WARNING("  ", i, " = ", req.a.at(i).s);
		req.a.at(i).s += "_0123456789";
	}
	ps->send(1005, req);
}

void TestProc(boost::shared_ptr<CbppSession> ps, StreamBuffer incoming){
	LOG_POSEIDON_WARNING("Received: ", HexDumper(incoming));
	TestMessage req(incoming);
	LOG_POSEIDON_WARNING("req.i = ", req.i);
	LOG_POSEIDON_WARNING("req.j = ", req.j);
	LOG_POSEIDON_WARNING("req.a.size() = ", req.a.size());
	for(std::size_t i = 0; i < req.a.size(); ++i){
		LOG_POSEIDON_WARNING("req.a[", i, "].s = ", req.a.at(i).s);
		LOG_POSEIDON_WARNING("req.a[", i, "].k = ", req.a.at(i).k);
	}
	ps->send(1006, req);
}

}

MODULE_RAII {
	return HttpServletDepository::create(1, SSLIT("/profile"), &profile_proc);
}
MODULE_RAII {
	return HttpServletDepository::create(1, SSLIT("/load"), &load_proc);
}
MODULE_RAII {
	return HttpServletDepository::create(1, SSLIT("/unload"), &unload_proc);
}
MODULE_RAII {
	AUTO(v, boost::make_shared<std::vector<boost::shared_ptr<void> > >());
	g_servlets = v;
	return v;
}
MODULE_RAII {
	return EventDispatcher::register_listener<TestEvent1>(&event1_proc);
}
MODULE_RAII {
	return EventDispatcher::register_listener<TestEvent2>(&event2_proc);
}
MODULE_RAII {
	return WebSocketServletDepository::create(2, SSLIT("/wstest"), &web_socket_proc);
}

MODULE_RAII {
	return CbppServletDepository::create(2, 100, &TestIntProc);
}
MODULE_RAII {
	return CbppServletDepository::create(2, 101, &TestUIntProc);
}
MODULE_RAII {
	return CbppServletDepository::create(2, 102, &TestStringProc);
}
MODULE_RAII {
	return CbppServletDepository::create(2, 103, &TestIntArrayProc);
}
MODULE_RAII {
	return CbppServletDepository::create(2, 104, &TestUIntArrayProc);
}
MODULE_RAII {
	return CbppServletDepository::create(2, 105, &TestStringArrayProc);
}
MODULE_RAII {
	return CbppServletDepository::create(2, 106, &TestProc);
}

namespace {

void on_client_close(int id){
	LOG_POSEIDON_FATAL("Client closed! ", id);
}

}

MODULE_RAII {

	LOG_POSEIDON_INFO("Connecting to github...");
	AUTO(p, TestClient::create());
	p->register_on_close(boost::bind(&on_client_close, 0));
	p->register_on_close(boost::bind(&on_client_close, 1));
	p->register_on_close(boost::bind(&on_client_close, 2));
	p->register_on_close(boost::bind(&on_client_close, 3));
	p->send(StreamBuffer("GET / HTTP/1.1\r\nHost: github.com\r\n\r\n"));

	AUTO(obj, boost::make_shared<MySqlObjF>());
	obj->set_si(999);
	obj->set_str("meow");
	obj->set_bi(456789);
	obj->async_save(false);

	TestMessage req;
	req.i = 12345;
	req.j = 54321;
	req.a.resize(2);
	req.a.at(0).s = "meow";
	req.a.at(0).k = 56789;
	std::memcpy(req.a.at(0).b, "ABC", 3);
	req.a.at(1).s = "bark";
	req.a.at(1).k = 98765;
	std::memcpy(req.a.at(1).b, "DEF", 3);
	LOG_POSEIDON_FATAL(req);
	return VAL_INIT;
}

MODULE_RAII {
	AUTO(server, (boost::make_shared<CbppServer>(2,
		IpPort(SharedNts("0.0.0.0"), 8850), NULLPTR, NULLPTR)));
	EpollDaemon::register_server(server);
	return server;
}
MODULE_RAII {
	AUTO(server, (boost::make_shared<HttpServer>(2,
		IpPort(SharedNts("0.0.0.0"), 8860), NULLPTR, NULLPTR, std::vector<std::string>())));
	EpollDaemon::register_server(server);
	return server;
}

MODULE_RAII {
	std::set<Uuid> s;
	for(unsigned i = 0; i < 1000000; ++i){
		s.insert(Uuid::generate());
	}
	LOG_POSEIDON_FATAL("number of uuid generated: ", s.size());
	LOG_POSEIDON_FATAL("first: ", *s.begin());
}

MODULE_RAII {
	LOG_POSEIDON_FATAL("----------- ", explode<std::string>(':', "0:1:2:3:").size());
	return VAL_INIT;
}

namespace {
	// const AUTO(sp, boost::make_shared<int>());
	boost::shared_ptr<int> sp;

	class MyJob : public JobBase {
	private:
		const char *const m_s;

	public:
		explicit MyJob(const char *s)
			: m_s(s)
		{
		}

	public:
		boost::weak_ptr<const void> get_category() const OVERRIDE {
			return sp;
		}
		void perform() const OVERRIDE {
			LOG_POSEIDON_FATAL("!!!! MyJob::perform: ", m_s);
			suspend_current_job(VAL_INIT);
		}
	};

	class MeowJob : public JobBase {
	public:
		boost::weak_ptr<const void> get_category() const OVERRIDE {
			return VAL_INIT;
		}
		void perform() const OVERRIDE {
			LOG_POSEIDON_FATAL("!!!! MeowJob::perform");
		}
	};
}

MODULE_RAII {
	AUTO(withdrawn, boost::make_shared<bool>(true));
	enqueue_job(boost::make_shared<MyJob>("1  "), 0, withdrawn);
	enqueue_job(boost::make_shared<MyJob>(" 2 "), 0, withdrawn);
	enqueue_job(boost::make_shared<MyJob>("  3"), 0, withdrawn);

	enqueue_job(boost::make_shared<MeowJob>());

	LOG_POSEIDON_FATAL("Job enqueued!");
	return VAL_INIT;
}

namespace {

void meow_proc(boost::shared_ptr<Http::Session> session, Http::Request req){
	PROFILE_ME;

	LOG_POSEIDON_FATAL("Contents = ", req.contents);

	OptionalMap headers;
	headers.set("Content-Type", "text/html");
	StreamBuffer contents;
	contents.put("<h1>Meow!</h1>");
	session->send(Http::ST_OK, STD_MOVE(headers), STD_MOVE(contents));
}

}

MODULE_RAII {
	AUTO(server, boost::make_shared<Http::Server>(2,
		IpPort(SharedNts("0.0.0.0"), 8860), NULLPTR, NULLPTR, std::vector<std::string>()));
	EpollDaemon::register_server(server);
	return server;
}
MODULE_RAII {
	return HttpServletDepository::create(2, SSLIT("/meow"), &meow_proc);
}


#include "../src/precompiled.hpp"
#include "../src/http/client.hpp"
#include "../src/http/utilities.hpp"
#include "../src/log.hpp"
#include "../src/module_raii.hpp"

namespace {

using namespace Poseidon;

class MyClient : public Http::Client {
public:
	static boost::shared_ptr<MyClient> create(){
		boost::shared_ptr<MyClient> ret(new MyClient);
		ret->go_resident();

		OptionalMap headers;
		headers.set("Host", "github.com");
//		headers.set("Transfer-Encoding", "chunked");
//		headers.set("Connection", "Close");
		ret->send(Http::V_GET, "/", OptionalMap(), STD_MOVE(headers), StreamBuffer("test"));
//		ret->shutdown_write();

		return ret;
	}

private:
	boost::uint64_t m_content_length;
	StreamBuffer m_contents;

private:
	MyClient()
		: Http::Client(IpPort(SSLIT("192.30.252.131"), 443), true)
	{
		LOG_POSEIDON_FATAL("MyClient::MyClient()");
	}

public:
	~MyClient(){
		LOG_POSEIDON_FATAL("MyClient::~MyClient()");
	}

private:
	void print() const {
		LOG_POSEIDON_INFO("Complete content:\n", m_contents.dump());
	}

protected:
	void on_response_headers(const Http::ResponseHeaders &response_headers, boost::uint64_t content_length) OVERRIDE {
		LOG_POSEIDON_DEBUG("on_response_headers(): status_code = ", static_cast<unsigned>(response_headers.status_code),
			", content_length = ", static_cast<boost::int64_t>(content_length));
		for(AUTO(it, response_headers.headers.begin()); it != response_headers.headers.end(); ++it){
			LOG_POSEIDON_DEBUG("> ", it->first, " = ", Http::url_decode(it->second));
		}
		m_content_length = content_length;
		m_contents.clear();
	}
	void on_entity(boost::uint64_t content_offset, const StreamBuffer &entity) OVERRIDE {
		LOG_POSEIDON_DEBUG("on_entity(): content_offset = ", content_offset, ", size = ", entity.size());
		AUTO(temp, entity);
		m_contents.splice(temp);
		if(m_contents.size() >= m_content_length){
			print();
		}
	}
	void on_chunked_trailer(boost::uint64_t real_content_length, const OptionalMap &headers) OVERRIDE {
		LOG_POSEIDON_DEBUG("on_chunked_trailer(): real_content_length = ", real_content_length);
		for(AUTO(it, headers.begin()); it != headers.end(); ++it){
			LOG_POSEIDON_DEBUG("> ", it->first, " = ", Http::url_decode(it->second));
		}
		print();
	}
	void on_content_eof(boost::uint64_t real_content_length) OVERRIDE {
		LOG_POSEIDON_DEBUG("on_content_eof(): real_content_length = ", real_content_length);
		print();
	}
};

MODULE_RAII(handles){
	handles.push(MyClient::create());
}

}



#include "../src/precompiled.hpp"
#include "../src/async_job.hpp"
#include "../src/job_promise.hpp"
#include "../src/log.hpp"
#include "../src/time.hpp"
#include "../src/module_raii.hpp"
#include "../src/mysql/object_base.hpp"
#include "../src/singletons/mysql_daemon.hpp"
#include "../src/singletons/timer_daemon.hpp"
#include "../src/singletons/job_dispatcher.hpp"

using namespace Poseidon;

#define MYSQL_OBJECT_NAME	MySqlObj
#define MYSQL_OBJECT_FIELDS	\
	FIELD_SMALLINT(si)	\
	FIELD_STRING(str)	\
	FIELD_BIGINT(bi)	\
	FIELD_DATETIME(dt)
#include "../src/mysql/object_generator.hpp"

namespace {
	class DelayedPromise : public JobPromise {
	public:
		static boost::shared_ptr<const DelayedPromise> create(boost::uint64_t delay){
			// auto ptr = boost::make_shared<DelayedPromise>();
			boost::shared_ptr<DelayedPromise> ptr(new DelayedPromise);
			ptr->m_timer = TimerDaemon::register_timer(delay, 0,
				std::bind([](boost::weak_ptr<DelayedPromise> weak){
					auto ptr = weak.lock();
					if(ptr){
						ptr->set_success();
					}
				}, boost::weak_ptr<DelayedPromise>(ptr)));
			return ptr;
		}

	private:
		boost::shared_ptr<const TimerItem> m_timer;

	private:
		DelayedPromise() = default;
	};
}

MODULE_RAII(handles){
	handles.push(TimerDaemon::register_timer(1000, 0,
		std::bind([]{
			try {
				AUTO(obj, boost::make_shared<MySqlObj>());
				obj->enable_auto_saving();
				obj->sync_load("SELECT * FROM `MySqlObj` LIMIT 1");
				LOG_POSEIDON_FATAL("Loaded: si = ", obj->get_si(),
					", str = ", obj->unlocked_get_str(), ", bi = ", obj->get_bi(), ", dt = ", obj->get_dt());
			} catch(std::exception &e){
				LOG_POSEIDON_FATAL("Exception: what = ", e.what());
			}
		})
	));

	enqueue_async_job([]{
		LOG_POSEIDON_FATAL("--- 1");
		JobDispatcher::yield(DelayedPromise::create(1000));

		LOG_POSEIDON_FATAL("--- 2");
		JobDispatcher::yield(DelayedPromise::create(2000));

		LOG_POSEIDON_FATAL("--- 3");
		JobDispatcher::yield(DelayedPromise::create(3000));

		LOG_POSEIDON_FATAL("--- 4");
		DEBUG_THROW(Exception, sslit("meow"));

		LOG_POSEIDON_FATAL("--- 5");
		JobDispatcher::yield(DelayedPromise::create(5000));
	});
	enqueue_async_job([]{
		LOG_POSEIDON_FATAL("+++ 1");
		JobDispatcher::yield(DelayedPromise::create(1000));

		LOG_POSEIDON_FATAL("+++ 2");
		JobDispatcher::yield(DelayedPromise::create(2000));

		LOG_POSEIDON_FATAL("+++ 3");
		JobDispatcher::yield(DelayedPromise::create(3000));

		LOG_POSEIDON_FATAL("+++ 4");
		DEBUG_THROW(Exception, sslit("meow"));

		LOG_POSEIDON_FATAL("+++ 5");
		JobDispatcher::yield(DelayedPromise::create(5000));
	});
	LOG_POSEIDON_FATAL("enqueued!");
}
*/

#include "../src/singletons/dns_daemon.hpp"
#include "../src/singletons/job_dispatcher.hpp"
#include "../src/async_job.hpp"
#include "../src/job_promise.hpp"
#include "../src/sock_addr.hpp"
#include "../src/ip_port.hpp"
#include "../src/log.hpp"
#include "../src/module_raii.hpp"
#include <boost/make_shared.hpp>

MODULE_RAII(){
	Poseidon::enqueue_async_job([]{
		auto sock_addr = boost::make_shared<Poseidon::SockAddr>();
		auto promise = Poseidon::DnsDaemon::async_lookup(sock_addr, "www.google.com", 80);
		Poseidon::JobDispatcher::yield(promise);
		promise->check_and_rethrow();
		LOG_POSEIDON_FATAL("Async result = ", Poseidon::get_ip_port_from_sock_addr(*sock_addr));
	});

	auto sock_addr = Poseidon::DnsDaemon::sync_lookup("www.google.com", 80);
	LOG_POSEIDON_FATAL("Sync result = ", Poseidon::get_ip_port_from_sock_addr(sock_addr));
}
