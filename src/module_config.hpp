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

#define DECLARE_MODULE_CONFIG(getConfig_, getConfigV_)	\
	namespace ModuleConfigImpl_ {	\
		extern ::boost::weak_ptr<const ::Poseidon::ConfigFile> g_weakConfig_;	\
		extern const char *getConfigFileName_();	\
		MODULE_RAII_PRIORITY(handles_, LONG_MIN){	\
			AUTO(config_, g_weakConfig_.lock());	\
			if(!config_){	\
				AUTO(newConfig_, ::boost::make_shared< ::Poseidon::ConfigFile>());	\
				newConfig_->load(getConfigFileName_());	\
				g_weakConfig_ = newConfig_;	\
				config_ = newConfig_;	\
			}	\
			handles_.push(STD_MOVE_IDN(config_));	\
		}	\
		inline ::boost::shared_ptr<const ::Poseidon::ConfigFile> requireConfig_(){	\
			const AUTO(config_, g_weakConfig_.lock());	\
			if(!config_){	\
				DEBUG_THROW(::Poseidon::Exception, SSLIT("Module config is not loaded"));	\
			}	\
			return config_;	\
		}	\
	}	\
	template<typename T_>	\
	bool getConfig_(T_ &val_, const char *key_){	\
		return ModuleConfigImpl_::requireConfig_()->get(val_, key_);	\
	}	\
	template<typename T_, typename DefaultT_>	\
	bool getConfig_(T_ &val_, const char *key_, const DefaultT_ &defVal_){	\
		return ModuleConfigImpl_::requireConfig_()->get(val_, key_, defVal_);	\
	}	\
	template<typename T_>	\
	T_ getConfig_(const char *key_){	\
		return ModuleConfigImpl_::requireConfig_()->get(key_);	\
	}	\
	template<typename T_, typename DefaultT_>	\
	T_ getConfig_(const char *key_, const DefaultT_ &defVal_){	\
		return ModuleConfigImpl_::requireConfig_()->get(key_, defVal_);	\
	}	\
	template<typename T_>	\
	std::size_t getConfigV_(std::vector<T_> &vals_, const char *key_, bool includingEmpty_ = false){	\
		return ModuleConfigImpl_::requireConfig_()->getAll(vals_, key_, includingEmpty_);	\
	}	\
	template<typename T_>	\
	std::vector<T_> getConfigV_(const char *key_, bool includingEmpty_ = false){	\
		return ModuleConfigImpl_::requireConfig_()->getAll(key_, includingEmpty_);	\
	}

#define DEFINE_MODULE_CONFIG(fileName_)	\
	namespace ModuleConfigImpl_ {	\
		::boost::weak_ptr<const ::Poseidon::ConfigFile> g_weakConfig_;	\
		const char *getConfigFileName_(){	\
			return fileName_;	\
		}	\
	}

#endif
