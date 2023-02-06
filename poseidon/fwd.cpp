// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "precompiled.ipp"
#include "fwd.hpp"
#include "static/main_config.hpp"
#include "static/fiber_scheduler.hpp"
#include "static/async_logger.hpp"
#include "static/timer_driver.hpp"
#include "static/async_task_executor.hpp"
#include "static/network_driver.hpp"

namespace poseidon {

atomic_signal exit_signal;
Main_Config& main_config = *new Main_Config;
Fiber_Scheduler& fiber_scheduler = *new Fiber_Scheduler;

Async_Logger& async_logger = *new Async_Logger;
Timer_Driver& timer_driver = *new Timer_Driver;
Async_Task_Executor& async_task_executor = *new Async_Task_Executor;
Network_Driver& network_driver = *new Network_Driver;

}  // namespace poseidon
