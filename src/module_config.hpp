// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_MODULE_CONFIG_HPP_
#define POSEIDON_MODULE_CONFIG_HPP_

#include "cxx_ver.hpp"
#include "config_file.hpp"
#include "module_raii.hpp"
#include "exception.hpp"
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <boost/make_shared.hpp>

// 这两个宏必须定义在同一个命名空间下。

#define DECLARE_MODULE_CONFIG(get_config_, get_config_v_)   \
	namespace ModuleConfigImpl_ {	\
		namespace {	\
			::boost::weak_ptr<const ::Poseidon::ConfigFile> g_weak_config_file_local_;	\
			inline ::boost::shared_ptr<const ::Poseidon::ConfigFile> require_config_file_(){	\
				const AUTO(config_file_, g_weak_config_file_local_.lock());	\
				if(!config_file_){   \
					DEBUG_THROW(::Poseidon::Exception, ::Poseidon::sslit("Module config is not loaded"));   \
				}   \
				return config_file_; \
			}	\
		}	\
		extern ::boost::shared_ptr<const ::Poseidon::ConfigFile> get_config_file_();	\
		MODULE_RAII_PRIORITY(handles_, LONG_MIN){	\
			AUTO(config_file_, get_config_file_());	\
			g_weak_config_file_local_ = config_file_;	\
			handles_.push(STD_MOVE_IDN(config_file_));	\
		}	\
	}   \
	template<typename T_>   \
	bool get_config_(T_ &val_, const char *key_){   \
		return ModuleConfigImpl_::require_config_file_()->get<T_>(val_, key_);   \
	}   \
	template<typename T_, typename DefaultT_>   \
	bool get_config_(T_ &val_, const char *key_, const DefaultT_ &def_val_){    \
		return ModuleConfigImpl_::require_config_file_()->get<T_, DefaultT_>(val_, key_, def_val_);  \
	}   \
	template<typename T_>   \
	T_ get_config_(const char *key_){   \
		return ModuleConfigImpl_::require_config_file_()->get<T_>(key_); \
	}   \
	template<typename T_, typename DefaultT_>   \
	T_ get_config_(const char *key_, const DefaultT_ &def_val_){    \
		return ModuleConfigImpl_::require_config_file_()->get<T_, DefaultT_>(key_, def_val_);    \
	}   \
	template<typename T_>   \
	std::size_t get_config_v_(std::vector<T_> &vals_, const char *key_, bool including_empty_ = false){ \
		return ModuleConfigImpl_::require_config_file_()->get_all<T_>(vals_, key_, including_empty_);    \
	}   \
	template<typename T_>   \
	std::vector<T_> get_config_v_(const char *key_, bool including_empty_ = false){ \
		return ModuleConfigImpl_::require_config_file_()->get_all<T_>(key_, including_empty_);   \
	}

#define DEFINE_MODULE_CONFIG(file_name_)    \
	namespace ModuleConfigImpl_ {   \
		::boost::weak_ptr< ::Poseidon::ConfigFile> g_weak_config_file_;	\
		::boost::shared_ptr<const ::Poseidon::ConfigFile> get_config_file_(){	\
			AUTO(config_file_, g_weak_config_file_.lock());   \
			if(!config_file_){   \
				config_file_ = ::boost::make_shared< ::Poseidon::ConfigFile>(); \
				config_file_->load(file_name_); \
				g_weak_config_file_ = config_file_;   \
			}	\
			return STD_MOVE_IDN(config_file_);	\
		}	\
	}

#endif
