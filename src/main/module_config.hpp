#ifndef POSEIDON_MODULE_CONFIG_HPP_
#define POSEIDON_MODULE_CONFIG_HPP_

#include "cxx_ver.hpp"
#include "config_file.hpp"
#include "module_raii.hpp"
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <boost/make_shared.hpp>

// 这两个宏必须定义在同一个命名空间下。

#define DECLARE_MODULE_CONFIG(func_)	\
	namespace ModuleConfigImpl_ {	\
		extern ::boost::weak_ptr<const ::Poseidon::ConfigFile> g_weakConfig_;	\
		extern const char *getConfigFileName_();	\
		MODULE_RAII {	\
			AUTO(config_, g_weakConfig_.lock());	\
			if(!config_){	\
				AUTO(newConfig_, ::boost::make_shared< ::Poseidon::ConfigFile>());	\
				newConfig_->load(getConfigFileName_());	\
				g_weakConfig_ = newConfig_;	\
				config_ = newConfig_;	\
			}	\
			return STD_MOVE_IDN(config_);	\
		}	\
	}	\
	inline ::boost::shared_ptr<const ::Poseidon::ConfigFile> func_ (){	\
		return ::boost::shared_ptr<const ::Poseidon::ConfigFile>(ModuleConfigImpl_::g_weakConfig_);	\
	}

#define DEFINE_MODULE_CONFIG(fileName_)	\
	namespace ModuleConfigImpl_ {	\
		::boost::weak_ptr<const ::Poseidon::ConfigFile> g_weakConfig_;	\
		const char *getConfigFileName_(){	\
			return fileName_;	\
		}	\
	}

#endif
