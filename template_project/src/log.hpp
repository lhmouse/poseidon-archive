#ifndef TEMPLATE_PROJECT_LOG_HPP_
#define TEMPLATE_PROJECT_LOG_HPP_

#include <poseidon/log.hpp>

namespace TemplateProject {

const unsigned long long LOG_CATEGORY = 0x00013100;

}

#define LOG_TEMPLATE_PROJECT(level_, ...)	\
	LOG_MASK(::TemplateProject::LOG_CATEGORY | (level_), __VA_ARGS__)

#define LOG_TEMPLATE_PROJECT_FATAL(...)        LOG_TEMPLATE_PROJECT(::Poseidon::Logger::LV_FATAL,     __VA_ARGS__)
#define LOG_TEMPLATE_PROJECT_ERROR(...)        LOG_TEMPLATE_PROJECT(::Poseidon::Logger::LV_ERROR,     __VA_ARGS__)
#define LOG_TEMPLATE_PROJECT_WARNING(...)      LOG_TEMPLATE_PROJECT(::Poseidon::Logger::LV_WARNING,   __VA_ARGS__)
#define LOG_TEMPLATE_PROJECT_INFO(...)         LOG_TEMPLATE_PROJECT(::Poseidon::Logger::LV_INFO,      __VA_ARGS__)
#define LOG_TEMPLATE_PROJECT_DEBUG(...)        LOG_TEMPLATE_PROJECT(::Poseidon::Logger::LV_DEBUG,     __VA_ARGS__)
#define LOG_TEMPLATE_PROJECT_TRACE(...)        LOG_TEMPLATE_PROJECT(::Poseidon::Logger::LV_TRACE,     __VA_ARGS__)

#endif
