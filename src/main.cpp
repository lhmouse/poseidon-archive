// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "singletons/main_config.hpp"
#include "singletons/dns_daemon.hpp"
#include "singletons/timer_daemon.hpp"
#include "singletons/mysql_daemon.hpp"
#include "singletons/mongodb_daemon.hpp"
#include "singletons/workhorse_camp.hpp"
#include "singletons/epoll_daemon.hpp"
#include "singletons/system_server.hpp"
#include "singletons/job_dispatcher.hpp"
#include "singletons/module_depository.hpp"
#include "singletons/event_dispatcher.hpp"
#include "singletons/filesystem_daemon.hpp"
#include "singletons/profile_depository.hpp"
#include "log.hpp"
#include "profiler.hpp"
#include "time.hpp"
#include "exception.hpp"
#include "atomic.hpp"
#include "checked_arithmetic.hpp"
#include "system_servlet_base.hpp"
#include "json.hpp"
#include <signal.h>

namespace Poseidon {

namespace {
	volatile bool g_running = true;

	void sigterm_proc(int){
		LOG_POSEIDON_WARNING("Received SIGTERM, will now exit...");
		atomic_store(g_running, false, ATOMIC_RELEASE);
	}

	void sighup_proc(int){
		LOG_POSEIDON_WARNING("Received SIGHUP, will now exit...");
		atomic_store(g_running, false, ATOMIC_RELEASE);
	}

	void sigint_proc(int){
		static const boost::uint64_t s_kill_timeout = 5000;
		static const boost::uint64_t s_reset_timeout = 10000;
		static boost::uint64_t s_kill_timer = 0;

		// 系统启动的时候这个时间是从 0 开始的，如果这时候按下 Ctrl+C 就会立即终止。
		// 因此将计时器的起点设为该区间以外。
		const AUTO(virtual_now, saturated_add(get_fast_mono_clock(), s_reset_timeout));
		if(saturated_sub(virtual_now, s_kill_timer) >= s_reset_timeout){
			s_kill_timer = virtual_now;
		}
		if(saturated_sub(virtual_now, s_kill_timer) >= s_kill_timeout){
			LOG_POSEIDON_FATAL("---------------------- Process is killed ----------------------");
			std::_Exit(EXIT_FAILURE);
		}
		LOG_POSEIDON_WARNING("Received SIGINT, trying to exit gracefully... If I don't terminate in ", s_kill_timeout, " milliseconds, press ^C again.");
		atomic_store(g_running, false, ATOMIC_RELEASE);
	}

	JsonObject make_help(const char *const (*param_info)[2]){
		JsonObject obj;
		for(AUTO(ptr, param_info); (*ptr)[0]; ++ptr){
			obj.set(SharedNts::view((*ptr)[0]), (*ptr)[1]);
		}
		return obj;
	}

	struct SystemServlet_help : public SystemServletBase {
		const char *get_uri() const FINAL {
			return "/poseidon/help";
		}
		void handle_get(JsonObject &resp) const FINAL {
			resp.set(sslit("description"), "Retreive general information about this process.");
			static const char *const PARAM_INFO[][2] = {
				{ NULLPTR }
			};
			resp.set(sslit("parameters"), make_help(PARAM_INFO));
		}
		void handle_post(JsonObject &resp, JsonObject /*req*/) const FINAL {
			// .servlets = list of servlets
			boost::container::vector<boost::shared_ptr<const SystemServletBase> > servlets;
			SystemServer::get_all_servlets(servlets);
			JsonArray arr;
			for(AUTO(it, servlets.begin()); it != servlets.end(); ++it){
				const AUTO_REF(servlet, *it);
				arr.push_back(servlet->get_uri());
			}
			resp.set(sslit("servlets"), STD_MOVE(arr));
		}
	};

	struct SystemServlet_logger : public SystemServletBase {
		const char *get_uri() const FINAL {
			return "/poseidon/logger";
		}
		void handle_get(JsonObject &resp) const FINAL {
			resp.set(sslit("description"), "Enable or disable specific levels of logs.");
			static const char *const PARAM_INFO[][2] = {
				{ "mask_to_disable",  "Log levels corresponding to bit ones here will be disabled.\n"
				                      "This parameter shall be a String of digit zeroes and ones.\n"
				                      "This parameter is overriden by `mask_to_enable`." },
				{ "mask_to_enable", "Log levels corresponding to bit ones here will be enabled.\n"
				                    "This parameter shall be a String of digit zeroes and ones.\n"
				                    "This parameter overrides `mask_to_disable`." },
				{ NULLPTR }
			};
			resp.set(sslit("parameters"), make_help(PARAM_INFO));
		}
		void handle_post(JsonObject &resp, JsonObject req) const FINAL {
			std::bitset<64> mask_to_enable, mask_to_disable;
			if(req.has("mask_to_enable")){
				try {
					mask_to_enable = std::bitset<64>(req.get("mask_to_enable").get<std::string>());
				} catch(std::exception &e){
					LOG_POSEIDON_WARNING("std::exception thrown: ", e.what());
					resp.set(sslit("error"), "Invalid parameter `mask_to_enable`: It shall be a String of digit zeroes and ones.");
					return;
				}
			}
			if(req.has("mask_to_disable")){
				try {
					mask_to_disable = std::bitset<64>(req.get("mask_to_disable").get<std::string>());
				} catch(std::exception &e){
					LOG_POSEIDON_WARNING("std::exception thrown: ", e.what());
					resp.set(sslit("error"), "Invalid parameter `mask_to_disable`: It shall be a String of digit zeroes and ones.");
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
			resp.set(sslit("mask_old"), mask_old.to_string());
			// .mask_new = current log mask
			resp.set(sslit("mask_new"), mask_new.to_string());
		}
	};

	struct SystemServlet_network : public SystemServletBase {
		const char *get_uri() const FINAL {
			return "/poseidon/network";
		}
		void handle_get(JsonObject &resp) const FINAL {
			resp.set(sslit("description"), "Retreive information about incoming and outgoing connections in this server process.");
			static const char *const PARAM_INFO[][2] = {
				{ NULLPTR }
			};
			resp.set(sslit("parameters"), make_help(PARAM_INFO));
		}
		void handle_post(JsonObject &resp, JsonObject /*req*/) const FINAL {
			// .sockets = all sockets managed by epoll.
			boost::container::vector<EpollDaemon::SnapshotElement> snapshot;
			EpollDaemon::snapshot(snapshot);
			JsonArray arr;
			char str[256];
			for(AUTO(it, snapshot.begin()); it != snapshot.end(); ++it){
				const AUTO_REF(elem, *it);
				JsonObject obj;
				obj.set(sslit("remote_info"), std::string(str, (unsigned)::snprintf(str, sizeof(str), "%s:%u", elem.remote_info.ip(), elem.remote_info.port())));
				obj.set(sslit("local_info"), std::string(str, (unsigned)::snprintf(str, sizeof(str), "%s:%u", elem.local_info.ip(), elem.local_info.port())));
				obj.set(sslit("creation_time"), std::string(str, format_time(str, sizeof(str), elem.creation_time, false)));
				obj.set(sslit("listening"), elem.listening);
				obj.set(sslit("readable"), elem.readable);
				obj.set(sslit("writeable"), elem.writeable);
				arr.push_back(STD_MOVE(obj));
			}
			resp.set(sslit("sockets"), STD_MOVE(arr));
		}
	};

	struct SystemServlet_profiler : public SystemServletBase {
		const char *get_uri() const FINAL {
			return "/poseidon/profiler";
		}
		void handle_get(JsonObject &resp) const FINAL {
			resp.set(sslit("description"), "View profiling information that has been collected within this process.");
			static const char *const PARAM_INFO[][2] = {
				{ "clear", "If set to `true`, all data will be purged." },
				{ NULLPTR }
			};
			resp.set(sslit("parameters"), make_help(PARAM_INFO));
		}
		void handle_post(JsonObject &resp, JsonObject req) const FINAL {
			bool clear = false;
			if(req.has("clear")){
				try {
					clear = req.get("clear").get<bool>();
				} catch(std::exception &e){
					LOG_POSEIDON_WARNING("std::exception thrown: ", e.what());
					resp.set(sslit("error"), "Invalid parameter `clear`: It shall be a Boolean.");
					return;
				}
			}

			if(clear){
				ProfileDepository::clear();
			}

			// .profile = all profile data.
			boost::container::vector<ProfileDepository::SnapshotElement> snapshot;
			ProfileDepository::snapshot(snapshot);
			JsonArray arr;
			for(AUTO(it, snapshot.begin()); it != snapshot.end(); ++it){
				const AUTO_REF(elem, *it);
				JsonObject obj;
				obj.set(sslit("file"), elem.file);
				obj.set(sslit("line"), elem.line);
				obj.set(sslit("func"), elem.func);
				obj.set(sslit("samples"), elem.samples);
				obj.set(sslit("total"), elem.total);
				obj.set(sslit("exclusive"), elem.exclusive);
				arr.push_back(STD_MOVE(obj));
			}
			resp.set(sslit("profile"), STD_MOVE(arr));
		}
	};

	struct SystemServlet_modules : public SystemServletBase {
		const char *get_uri() const FINAL {
			return "/poseidon/modules";
		}
		void handle_get(JsonObject &resp) const FINAL {
			resp.set(sslit("description"), "Load or unload modules in the current process.");
			static const char *const PARAM_INFO[][2] = {
				{ "path_to_load", "The path to a shared object file which will be loaded.\n"
				                  "This path will be passed to `dlopen()`, hence the normal library search rules apply.\n"
				                  "This parameter cannot be specified together with `address_to_unload`." },
				{ "address_to_unload", "The module whose base address equals this value will be unloaded.\n"
				                       "This shall be a string representing a number in decimal, or hexadecimal with the prefix `0x`.\n"
				                       "This parameter cannot be specified together with `path_to_load`." },
				{ NULLPTR }
			};
			resp.set(sslit("parameters"), make_help(PARAM_INFO));
		}
		void handle_post(JsonObject &resp, JsonObject req) const FINAL {
			bool to_load = false;
			std::string path_to_load;
			if(req.has("path_to_load")){
				try {
					path_to_load = req.get("path_to_load").get<std::string>();
					to_load = true;
				} catch(std::exception &e){
					LOG_POSEIDON_WARNING("std::exception thrown: ", e.what());
					resp.set(sslit("error"), "Invalid parameter `path_to_load`: It shall be a String designating a shared object file to load.");
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
					LOG_POSEIDON_WARNING("std::exception thrown: ", e.what());
					resp.set(sslit("error"), "Invalid parameter `address_to_unload`: It shall be a String representing a number in decimal, or hexadecimal with the prefix `0x`.");
					return;
				}
			}
			if(to_load && to_unload){
				LOG_POSEIDON_WARNING("`address_to_load` and `path_to_unload` cannot be specified together.");
				resp.set(sslit("error"), "`address_to_load` and `path_to_unload` cannot be specified together.");
				return;
			}

			if(to_load){
				std::string what;
				try {
					ModuleDepository::load(path_to_load);
					// what.clear();
				} catch(std::exception &e){
					LOG_POSEIDON_WARNING("std::exception thrown: ", e.what());
					what = e.what();
				} catch(...){
					LOG_POSEIDON_WARNING("Unknown exception thrown.");
					what = "Unknown exception";
				}
				if(!what.empty()){
					LOG_POSEIDON_WARNING("Failed to load module: ", what);
					resp.set(sslit("error"), "Failed to load module: " + what);
					return;
				}
			}
			if(to_unload){
				const bool result = ModuleDepository::unload(address_to_unload);
				if(!result){
					LOG_POSEIDON_WARNING("Failed to unload module: 0x", std::hex, address_to_unload);
					resp.set(sslit("error"), "Failed to unload module. Maybe it has been unloaded already?");
					return;
				}
			}

			// .modules = all loaded modules.
			boost::container::vector<ModuleDepository::SnapshotElement> snapshot;
			ModuleDepository::snapshot(snapshot);
			JsonArray arr;
			char str[256];
			for(AUTO(it, snapshot.begin()); it != snapshot.end(); ++it){
				const AUTO_REF(elem, *it);
				JsonObject obj;
				obj.set(sslit("dl_handle"), std::string(str, (unsigned)::snprintf(str, sizeof(str), "0x%llx", (unsigned long long)elem.dl_handle)));
				obj.set(sslit("base_address"), std::string(str, (unsigned)::snprintf(str, sizeof(str), "0x%llx", (unsigned long long)elem.base_address)));
				obj.set(sslit("real_path"), elem.real_path.get());
				arr.push_back(STD_MOVE(obj));
			}
			resp.set(sslit("modules"), STD_MOVE(arr));
		}
	};

	template<typename T>
	struct RaiiSingletonRunner : NONCOPYABLE {
		RaiiSingletonRunner(){
			T::start();
		}
		~RaiiSingletonRunner(){
			T::stop();
		}
	};

#define START(x_)   const RaiiSingletonRunner<x_> UNIQUE_ID

	void run(){
		PROFILE_ME;

		START(DnsDaemon);
		START(FileSystemDaemon);
		START(MySqlDaemon);
		START(MongoDbDaemon);
		START(JobDispatcher);
		START(WorkhorseCamp);

		try {
			START(ModuleDepository);
			START(TimerDaemon);
			START(EpollDaemon);
			START(EventDispatcher);
			START(SystemServer);

			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Setting up built-in system servlets...");
			boost::container::vector<boost::shared_ptr<const SystemServletBase> > system_servlets;
			system_servlets.push_back(SystemServer::register_servlet(boost::make_shared<SystemServlet_help>()));
			system_servlets.push_back(SystemServer::register_servlet(boost::make_shared<SystemServlet_logger>()));
			system_servlets.push_back(SystemServer::register_servlet(boost::make_shared<SystemServlet_network>()));
			system_servlets.push_back(SystemServer::register_servlet(boost::make_shared<SystemServlet_profiler>()));
			system_servlets.push_back(SystemServer::register_servlet(boost::make_shared<SystemServlet_modules>()));

			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Waiting for daemon initialization to complete...");
			::timespec req;
			req.tv_sec = 0;
			req.tv_nsec = 200000000;
			::nanosleep(&req, NULLPTR);

			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Setting new log mask...");
			Logger::initialize_mask_from_config();

			const AUTO(init_modules, MainConfig::get_all<std::string>("init_module"));
			for(AUTO(it, init_modules.begin()); it != init_modules.end(); ++it){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Loading init module: ", *it);
				ModuleDepository::load(it->c_str());
			}

			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Waiting for all asynchronous MySQL operations to complete...");
			MySqlDaemon::wait_for_all_async_operations();
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Waiting for all asynchronous MongoDB operations to complete...");
			MongoDbDaemon::wait_for_all_async_operations();

			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Entering modal loop...");
			JobDispatcher::do_modal(g_running);

			Logger::finalize_mask();
		} catch(...){
			Logger::finalize_mask();
			throw;
		}
	}
}

}

using namespace Poseidon;

int main(int argc, char **argv)
try {
	Logger::set_thread_tag("P   "); // Primary

	::signal(SIGHUP, &sighup_proc);
	::signal(SIGTERM, &sigterm_proc);
	::signal(SIGINT, &sigint_proc);
	::signal(SIGCHLD, SIG_IGN);
	::signal(SIGPIPE, SIG_IGN);

	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "------------------------- Starting up -------------------------");

	const char *run_path;
	if(argc > 1){
		run_path = argv[1];
	} else{
		run_path = "/usr/etc/poseidon";
	}
	MainConfig::set_run_path(run_path);
	MainConfig::reload();

	START(ProfileDepository);
	run();

	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "------------------ Process exited gracefully ------------------");
	return EXIT_SUCCESS;
} catch(std::exception &e){
	LOG_POSEIDON_ERROR("std::exception thrown in main(): what = ", e.what());

	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_ERROR, "------------------ Process exited abnormally ------------------");
	return EXIT_FAILURE;
} catch(...){
	LOG_POSEIDON_ERROR("Unknown exception thrown in main().");

	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_ERROR, "------------------ Process exited abnormally ------------------");
	return EXIT_FAILURE;
}
