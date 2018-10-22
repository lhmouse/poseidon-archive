// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "fwd.hpp"
#include "singletons/main_config.hpp"
#include "singletons/dns_daemon.hpp"
#include "singletons/timer_daemon.hpp"
#include "singletons/workhorse_camp.hpp"
#include "singletons/epoll_daemon.hpp"
#include "singletons/system_http_server.hpp"
#include "singletons/job_dispatcher.hpp"
#include "singletons/module_depository.hpp"
#include "singletons/event_dispatcher.hpp"
#include "singletons/filesystem_daemon.hpp"
#include "singletons/profile_depository.hpp"
#include "singletons/simple_http_client_daemon.hpp"
#ifdef POSEIDON_ENABLE_MYSQL
#  include "singletons/mysql_daemon.hpp"
#endif
#ifdef POSEIDON_ENABLE_MONGODB
#  include "singletons/mongodb_daemon.hpp"
#endif
#ifdef POSEIDON_ENABLE_MAGIC
#  include "singletons/magic_daemon.hpp"
#endif
#include "log.hpp"
#include "profiler.hpp"
#include "time.hpp"
#include "exception.hpp"
#include "atomic.hpp"
#include "checked_arithmetic.hpp"
#include "system_http_servlet_base.hpp"
#include "json.hpp"
#include <signal.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>

using namespace Poseidon;

namespace {
	volatile int g_sig_recv = 0;

	void sig_proc(int sig){
		if(sig == SIGINT){
			static CONSTEXPR const boost::uint64_t s_kill_timeout = 5000;
			static CONSTEXPR const boost::uint64_t s_reset_timeout = 10000;
			static volatile boost::uint64_t s_kill_timer = 0;
			// 系统启动的时候这个时间是从 0 开始的，如果这时候按下 Ctrl+C 就会立即终止。
			// 因此将计时器的起点调整为该区间以外。
			const AUTO(virtual_now, saturated_add(get_fast_mono_clock(), s_reset_timeout));
			AUTO(delta, saturated_sub(virtual_now, atomic_load(s_kill_timer, memory_order_relaxed)));
			if(delta >= s_reset_timeout){
				atomic_store(s_kill_timer, virtual_now, memory_order_relaxed);
				delta = 0;
			}
			if(delta >= s_kill_timeout){
				::raise(SIGQUIT);
			}
		}
		atomic_store(g_sig_recv, sig, memory_order_release);
	}

	Json_object make_help(const char *const (*param_info)[2]){
		Json_object obj;
		for(AUTO(ptr, param_info); (*ptr)[0]; ++ptr){
			obj.set(Rcnts::view((*ptr)[0]), (*ptr)[1]);
		}
		return obj;
	}

	struct System_http_servlet_help : public System_http_servlet_base {
		const char * get_uri() const FINAL {
			return "/poseidon/help";
		}
		void handle_get(Json_object &resp) const FINAL {
			resp.set(Rcnts::view("description"), "Retreive general information about this process.");
			static const char *const s_param_info[][2] = {
				{ NULLPTR }
			};
			resp.set(Rcnts::view("parameters"), make_help(s_param_info));
		}
		void handle_post(Json_object &resp, Json_object /*req*/) const FINAL {
			// .servlets = list of servlets
			boost::container::vector<boost::shared_ptr<const System_http_servlet_base> > servlets;
			System_http_server::get_all_servlets(servlets);
			Json_array arr;
			for(AUTO(it, servlets.begin()); it != servlets.end(); ++it){
				const AUTO_REF(servlet, *it);
				arr.push_back(servlet->get_uri());
			}
			resp.set(Rcnts::view("servlets"), STD_MOVE_IDN(arr));
		}
	};

	struct System_http_servlet_logger : public System_http_servlet_base {
		const char * get_uri() const FINAL {
			return "/poseidon/logger";
		}
		void handle_get(Json_object &resp) const FINAL {
			resp.set(Rcnts::view("description"), "Enable or disable specific levels of logs.");
			static const char *const s_param_info[][2] = {
				{ "mask_to_disable",  "Log levels corresponding to bit ones here will be disabled.\n"
				                      "This parameter shall be a `String` of digit zeroes and ones.\n"
				                      "This parameter is overriden by `mask_to_enable`." },
				{ "mask_to_enable", "Log levels corresponding to bit ones here will be enabled.\n"
				                    "This parameter shall be a `String` of digit zeroes and ones.\n"
				                    "This parameter overrides `mask_to_disable`." },
				{ NULLPTR }
			};
			resp.set(Rcnts::view("parameters"), make_help(s_param_info));
		}
		void handle_post(Json_object &resp, Json_object req) const FINAL {
			std::bitset<64> mask_to_enable, mask_to_disable;
			if(req.has("mask_to_enable")){
				try {
					mask_to_enable = std::bitset<64>(req.get("mask_to_enable").get<std::string>());
				} catch(std::exception &e){
					POSEIDON_LOG_WARNING("std::exception thrown: ", e.what());
					resp.set(Rcnts::view("error"), "Invalid parameter `mask_to_enable`: It shall be a `String` of digit zeroes and ones.");
					return;
				}
			}
			if(req.has("mask_to_disable")){
				try {
					mask_to_disable = std::bitset<64>(req.get("mask_to_disable").get<std::string>());
				} catch(std::exception &e){
					POSEIDON_LOG_WARNING("std::exception thrown: ", e.what());
					resp.set(Rcnts::view("error"), "Invalid parameter `mask_to_disable`: It shall be a `String` of digit zeroes and ones.");
					return;
				}
			}

			std::bitset<64> mask_old, mask_new;
#ifdef POSEIDON_CXX11
			mask_old = Logger::set_mask(mask_to_disable.to_ullong(), mask_to_enable.to_ullong());
#else
			mask_old = Logger::set_mask(mask_to_disable.to_ulong(), mask_to_enable.to_ulong());
#endif
			mask_new = Logger::get_mask();
			// .mask_old = previous log mask
			resp.set(Rcnts::view("mask_old"), mask_old.to_string());
			// .mask_new = current log mask
			resp.set(Rcnts::view("mask_new"), mask_new.to_string());
		}
	};

	struct System_http_servlet_network : public System_http_servlet_base {
		const char * get_uri() const FINAL {
			return "/poseidon/network";
		}
		void handle_get(Json_object &resp) const FINAL {
			resp.set(Rcnts::view("description"), "Retreive information about incoming and outgoing connections in this process.");
			static const char *const s_param_info[][2] = {
				{ NULLPTR }
			};
			resp.set(Rcnts::view("parameters"), make_help(s_param_info));
		}
		void handle_post(Json_object &resp, Json_object /*req*/) const FINAL {
			// .sockets = all sockets managed by epoll.
			boost::container::vector<Epoll_daemon::Snapshot_element> snapshot;
			Epoll_daemon::snapshot(snapshot);
			Json_array arr;
			char str[256];
			for(AUTO(it, snapshot.begin()); it != snapshot.end(); ++it){
				const AUTO_REF(elem, *it);
				Json_object obj;
				obj.set(Rcnts::view("remote_info"), std::string(str, (unsigned)::snprintf(str, sizeof(str), "%s:%u", elem.remote_info.ip(), elem.remote_info.port())));
				obj.set(Rcnts::view("local_info"), std::string(str, (unsigned)::snprintf(str, sizeof(str), "%s:%u", elem.local_info.ip(), elem.local_info.port())));
				obj.set(Rcnts::view("creation_time"), std::string(str, format_time(str, sizeof(str), elem.creation_time, false)));
				obj.set(Rcnts::view("listening"), elem.listening);
				obj.set(Rcnts::view("readable"), elem.readable);
				obj.set(Rcnts::view("writable"), elem.writable);
				arr.push_back(STD_MOVE_IDN(obj));
			}
			resp.set(Rcnts::view("sockets"), STD_MOVE_IDN(arr));
		}
	};

	struct System_http_servlet_profiler : public System_http_servlet_base {
		const char * get_uri() const FINAL {
			return "/poseidon/profiler";
		}
		void handle_get(Json_object &resp) const FINAL {
			resp.set(Rcnts::view("description"), "View profiling information that has been collected within this process.");
			static const char *const s_param_info[][2] = {
				{ "clear", "If set to `true`, all data will be purged." },
				{ NULLPTR }
			};
			resp.set(Rcnts::view("parameters"), make_help(s_param_info));
		}
		void handle_post(Json_object &resp, Json_object req) const FINAL {
			bool clear = false;
			if(req.has("clear")){
				try {
					clear = req.get("clear").get<bool>();
				} catch(std::exception &e){
					POSEIDON_LOG_WARNING("std::exception thrown: ", e.what());
					resp.set(Rcnts::view("error"), "Invalid parameter `clear`: It shall be a `Boolean`.");
					return;
				}
			}

			if(clear){
				Profile_depository::clear();
			}

			// .profile = all profile data.
			boost::container::vector<Profile_depository::Snapshot_element> snapshot;
			Profile_depository::snapshot(snapshot);
			Json_array arr;
			for(AUTO(it, snapshot.begin()); it != snapshot.end(); ++it){
				const AUTO_REF(elem, *it);
				Json_object obj;
				obj.set(Rcnts::view("file"), elem.file);
				obj.set(Rcnts::view("line"), elem.line);
				obj.set(Rcnts::view("func"), elem.func);
				obj.set(Rcnts::view("samples"), elem.samples);
				obj.set(Rcnts::view("total"), elem.total);
				obj.set(Rcnts::view("exclusive"), elem.exclusive);
				arr.push_back(STD_MOVE_IDN(obj));
			}
			resp.set(Rcnts::view("profile"), STD_MOVE_IDN(arr));
		}
	};

	struct System_http_servlet_modules : public System_http_servlet_base {
		const char * get_uri() const FINAL {
			return "/poseidon/modules";
		}
		void handle_get(Json_object &resp) const FINAL {
			resp.set(Rcnts::view("description"), "Load or unload modules in the current process.");
			static const char *const s_param_info[][2] = {
				{ "path_to_load", "The path to a shared object file which will be loaded.\n"
				                  "This path will be passed to `dlopen()`, hence the normal library search rules apply.\n"
				                  "This parameter cannot be specified together with `address_to_unload`." },
				{ "address_to_unload", "The module whose base address equals this value will be unloaded.\n"
				                       "This shall be a string representing a number in decimal, or hexadecimal with the prefix `0x`.\n"
				                       "This parameter cannot be specified together with `path_to_load`." },
				{ NULLPTR }
			};
			resp.set(Rcnts::view("parameters"), make_help(s_param_info));
		}
		void handle_post(Json_object &resp, Json_object req) const FINAL {
			bool to_load = false;
			std::string path_to_load;
			if(req.has("path_to_load")){
				try {
					path_to_load = req.get("path_to_load").get<std::string>();
					to_load = true;
				} catch(std::exception &e){
					POSEIDON_LOG_WARNING("std::exception thrown: ", e.what());
					resp.set(Rcnts::view("error"), "Invalid parameter `path_to_load`: It shall be a `String` representing the path of a shared object file to load.");
					return;
				}
			}
			bool to_unload = false;
			void *address_to_unload = NULLPTR;
			if(req.has("address_to_unload")){
				try {
					const AUTO_REF(str, req.get("address_to_unload").get<std::string>());
					char *eptr;
					const AUTO(val, ::strtoull(str.c_str(), &eptr, 0));
					if(*eptr != 0){
						throw std::bad_cast(); // XXX
					}
#ifdef POSEIDON_CXX11
					address_to_unload = reinterpret_cast<void *>(boost::numeric_cast<std::uintptr_t>(val));
#else
					address_to_unload = reinterpret_cast<void *>(val);
#endif
					to_unload = true;
				} catch(std::exception &e){
					POSEIDON_LOG_WARNING("std::exception thrown: ", e.what());
					resp.set(Rcnts::view("error"), "Invalid parameter `address_to_unload`: It shall be a `String` representing a number in decimal, or hexadecimal with the prefix `0x`.");
					return;
				}
			}
			if(to_load && to_unload){
				POSEIDON_LOG_WARNING("`address_to_load` and `path_to_unload` cannot be specified together.");
				resp.set(Rcnts::view("error"), "`address_to_load` and `path_to_unload` cannot be specified together.");
				return;
			}

			if(to_load){
				std::string what;
				try {
					Module_depository::load(path_to_load);
					// what.clear();
				} catch(std::exception &e){
					POSEIDON_LOG_WARNING("std::exception thrown: ", e.what());
					what = e.what();
				} catch(...){
					POSEIDON_LOG_WARNING("Unknown exception thrown.");
					what = "Unknown exception";
				}
				if(!what.empty()){
					POSEIDON_LOG_WARNING("Failed to load module: ", what);
					resp.set(Rcnts::view("error"), "Failed to load module: " + what);
					return;
				}
			}
			if(to_unload){
				const bool result = Module_depository::unload(address_to_unload);
				if(!result){
					POSEIDON_LOG_WARNING("Failed to unload module: 0x", std::hex, address_to_unload);
					resp.set(Rcnts::view("error"), "Failed to unload module. Maybe it has been unloaded already?");
					return;
				}
			}

			// .modules = all loaded modules.
			boost::container::vector<Module_depository::Snapshot_element> snapshot;
			Module_depository::snapshot(snapshot);
			Json_array arr;
			char str[256];
			for(AUTO(it, snapshot.begin()); it != snapshot.end(); ++it){
				const AUTO_REF(elem, *it);
				Json_object obj;
				obj.set(Rcnts::view("dl_handle"), std::string(str, (unsigned)::snprintf(str, sizeof(str), "0x%llx", (unsigned long long)elem.dl_handle)));
				obj.set(Rcnts::view("base_address"), std::string(str, (unsigned)::snprintf(str, sizeof(str), "0x%llx", (unsigned long long)elem.base_address)));
				obj.set(Rcnts::view("real_path"), elem.real_path.get());
				arr.push_back(STD_MOVE_IDN(obj));
			}
			resp.set(Rcnts::view("modules"), STD_MOVE_IDN(arr));
		}
	};

	template<typename T>
	struct Raii_singleton_runner : NONCOPYABLE {
		Raii_singleton_runner(){
			T::start();
		}
		~Raii_singleton_runner(){
			T::stop();
		}
	};
}

int main(int argc, char **argv, char **/*envp*/){
	bool all_logs = false;
	bool daemonize = false;
	int help = 0; // 1 = exit with EXIT_SUCCESS, * = exit with EXIT_FAILURE.
	bool version = false;
	const char *new_wd = NULLPTR;

	int opt;
	while((opt = ::getopt(argc, argv, "adhv")) != -1){
		switch(opt){
		case 'a':
			all_logs = true;
			break;
		case 'd':
			daemonize = true;
			break;
		case 'h':
			help |= 1;
			break;
		case 'v':
			version = true;
			break;
		default:
			help |= 2;
			break;
		}
	}
	switch(argc - optind){
	case 0:
		break;
	case 1:
		new_wd = argv[optind];
		break;
	default:
		::fprintf(stderr, "%s: too many arguments -- '%s'\n", argv[0], argv[optind + 1]);
		help |= 2;
		break;
	}
	if(help){
		::fprintf(stdout,
            //        1         2         3         4         5         6         7         8
			// 345678901234567890123456789012345678901234567890123456789012345678901234567890
			"Usage: %s [-adhv] [<directory>]\n"
			"  -a            do not load `log_masked_levels` from 'main.conf'\n"
			"  -d            daemonize\n"
			"  -h            show this help message then exit\n"
			"  -v            print version string then exit\n"
			"  <directory>   set working directory here before anything else\n"
			, argv[0]);
		return (help == 1) ? EXIT_SUCCESS : EXIT_FAILURE;
	}
	if(version){
		::fprintf(stdout,
			"%s (built on %s %s)\n"
			"\n"
			"Visit the home page at <%s>.\n"
			"Report bugs to <%s>.\n"
			, PACKAGE_STRING, __DATE__, __TIME__, PACKAGE_URL, PACKAGE_BUGREPORT);
		return EXIT_SUCCESS;
	}

	if(daemonize && (::daemon(false, true) != 0)){
		const int err_code = errno;
		::fprintf(stderr, "%s: daemonization failed with %d (%s)", argv[0], err_code, ::strerror(err_code));
		return EXIT_FAILURE;
	}

	Logger::set_thread_tag("P   "); // Primary

	::signal(SIGHUP, &sig_proc);
	::signal(SIGTERM, &sig_proc);
	::signal(SIGINT, &sig_proc);
	::signal(SIGCHLD, SIG_IGN);
	::signal(SIGPIPE, SIG_IGN);

	POSEIDON_LOG(Logger::special_major | Logger::level_info, "Starting up: ", PACKAGE_STRING, " (built on ", __DATE__, " ", __TIME__, ")");
	try {
		if(new_wd){
			Main_config::set_run_path(new_wd);
		}
		Main_config::reload();

#define START(x_)   const Raii_singleton_runner<x_> POSEIDON_UNIQUE_NAME

		START(Profile_depository);
#ifdef POSEIDON_ENABLE_MAGIC
		START(Magic_daemon);
#endif
		START(Dns_daemon);
		START(Filesystem_daemon);
#ifdef POSEIDON_ENABLE_MYSQL
		START(Mysql_daemon);
#endif
#ifdef POSEIDON_ENABLE_MONGODB
		START(Mongodb_daemon);
#endif
		START(Job_dispatcher);
		START(Workhorse_camp);

		START(Module_depository);
		START(Timer_daemon);
		START(Epoll_daemon);
		START(Event_dispatcher);
		START(System_http_server);
		START(Simple_http_client_daemon);

		POSEIDON_LOG(Logger::special_major | Logger::level_info, "Setting up built-in system servlets...");
		boost::container::vector<boost::shared_ptr<const System_http_servlet_base> > system_http_servlets;
		system_http_servlets.push_back(System_http_server::register_servlet(boost::make_shared<System_http_servlet_help>()));
		system_http_servlets.push_back(System_http_server::register_servlet(boost::make_shared<System_http_servlet_logger>()));
		system_http_servlets.push_back(System_http_server::register_servlet(boost::make_shared<System_http_servlet_network>()));
		system_http_servlets.push_back(System_http_server::register_servlet(boost::make_shared<System_http_servlet_profiler>()));
		system_http_servlets.push_back(System_http_server::register_servlet(boost::make_shared<System_http_servlet_modules>()));

		if(!all_logs){
			POSEIDON_LOG(Logger::special_major | Logger::level_info, "Setting new log mask...");
			Logger::initialize_mask_from_config();
		}

		const AUTO(init_modules, Main_config::get_all<std::string>("init_module"));
		for(AUTO(it, init_modules.begin()); it != init_modules.end(); ++it){
			const AUTO(path, it->c_str());
			POSEIDON_LOG(Logger::special_major | Logger::level_info, "Loading init module: ", path);
			Module_depository::load(path);
		}

#ifdef POSEIDON_ENABLE_MYSQL
		POSEIDON_LOG(Logger::special_major | Logger::level_info, "Waiting for all asynchronous MySQL operations to complete...");
		Mysql_daemon::wait_for_all_async_operations();
#endif
#ifdef POSEIDON_ENABLE_MONGODB
		POSEIDON_LOG(Logger::special_major | Logger::level_info, "Waiting for all asynchronous MongoDB operations to complete...");
		Mongodb_daemon::wait_for_all_async_operations();
#endif

		POSEIDON_LOG(Logger::special_major | Logger::level_info, "Entering modal loop...");
		Job_dispatcher::do_modal(g_sig_recv);
	} catch(std::exception &e){
		Logger::finalize_mask();
		POSEIDON_LOG_ERROR("std::exception thrown in main(): what = ", e.what());
		return EXIT_FAILURE;
	} catch(...){
		Logger::finalize_mask();
		POSEIDON_LOG_ERROR("Unknown exception thrown in main().");
		return EXIT_FAILURE;
	}
	Logger::finalize_mask();
	POSEIDON_LOG(Logger::special_major | Logger::level_info, "Shutting down: ", PACKAGE_STRING, " (built on ", __DATE__, " ", __TIME__, ")");
	return EXIT_SUCCESS;
}
