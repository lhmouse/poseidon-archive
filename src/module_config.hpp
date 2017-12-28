// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_MODULE_CONFIG_HPP_
#define POSEIDON_MODULE_CONFIG_HPP_

#include "cxx_ver.hpp"
#include "cxx_util.hpp"
#include "config_file.hpp"
#include "module_raii.hpp"
#include "exception.hpp"
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <boost/make_shared.hpp>

/*===========================================================================*\

                    ---=* 模块（动态库）配置文件说明 *=---

DECLARE_MODULE_CONFIG(get_config) 展开后生成以下接口：

  bool get_config_raw(std::string &val, const char *key);
  const std::string &get_config_raw(const char *key);

  std::size_t get_config_all_raw(boost::container::vector<std::string> &vals,
      const char *key, bool including_empty = false);
  boost::container::vector<std::string> get_config_all_raw(
      const char *key, bool including_empty = false);

  template<typename T>
  bool get_config(T &val, const char *key);
  template<typename T>
  T get_config(const char *key);

  template<typename T, typename DefaultT>
  bool get_config(T &val, const char *key, const DefaultT &def_val);
  template<typename T, typename DefaultT>
  T get_config(const char *key, const DefaultT &def_val);

  template<typename T>
  std::size_t get_config_all(boost::container::vector<T> &vals,
      const char *key, bool including_empty = false);
  template<typename T>
  boost::container::vector<T> get_config_all(
      const char *key, bool including_empty = false);

DEFINE_MODULE_CONFIG(path) 在模块被加载时解析指定的配置文件。
  这个宏必须和对应的 DECLARE_MODULE_CONFIG 在同一个 namespace 中展开。

\*===========================================================================*/

#define DECLARE_MODULE_CONFIG(prefix_)	\
	/* Declare an access function for this module. This function is going to be defined in `DEFINE_MODULE_CONFIG`. */	\
	extern ::boost::shared_ptr<const ::Poseidon::ConfigFile> module_config_require_nifty_();	\
	/* Define getters with the suffix `_raw`. */	\
	inline bool TOKEN_CAT2(prefix_, _raw) (::std::string &val_, const char *key_){	\
		return module_config_require_nifty_()->get_raw(val_, key_);	\
	}	\
	inline const ::std::string & TOKEN_CAT2(prefix_, _raw) (const char *key_){	\
		return module_config_require_nifty_()->get_raw(key_);	\
	}	\
	/* Define getters with the suffix `_all_raw`. */	\
	inline ::std::size_t TOKEN_CAT2(prefix_, _all_raw) (::boost::container::vector< ::std::string> &vals_, const char *key_, bool including_empty_ = false){	\
		return module_config_require_nifty_()->get_all_raw(vals_, key_, including_empty_);	\
	}	\
	inline ::boost::container::vector< ::std::string> TOKEN_CAT2(prefix_, _all_raw) (const char *key_, bool including_empty_ = false){	\
		return module_config_require_nifty_()->get_all_raw(key_, including_empty_);	\
	}	\
	/* Define getters with no suffix and performs value-initialization if no entry is found. */	\
	template<typename T_>	\
	inline bool TOKEN_CAT2(prefix_, ) (T_ &val_, const char *key_){	\
		return module_config_require_nifty_()->get<T_>(val_, key_);	\
	}	\
	template<typename T_>	\
	inline T_ TOKEN_CAT2(prefix_, ) (const char *key_){	\
		return module_config_require_nifty_()->get<T_>(key_);	\
	}	\
	/* Define getters with no suffix and requires the user to provide a default value if no entry is found. */	\
	template<typename T_, typename DefValT_>	\
	inline bool TOKEN_CAT2(prefix_, ) (T_ &val_, const char *key_, const DefValT_ &def_val_){	\
		return module_config_require_nifty_()->get<T_>(val_, key_, def_val_);	\
	}	\
	template<typename T_, typename DefValT_>	\
	inline T_ TOKEN_CAT2(prefix_, ) (const char *key_, const DefValT_ &def_val_){	\
		return module_config_require_nifty_()->get<T_>(key_, def_val_);	\
	}	\
	/* Define getters with the suffix `_all`. */	\
	template<typename T_>	\
	inline ::std::size_t TOKEN_CAT2(prefix_, _all) (::boost::container::vector<T_> &vals_, const char *key_, bool including_empty_ = false){	\
		return module_config_require_nifty_()->get_all<T_>(vals_, key_, including_empty_);	\
	}	\
	template<typename T_>	\
	inline ::boost::container::vector<T_> TOKEN_CAT2(prefix_, _all) (const char *key_, bool including_empty_ = false){	\
		return module_config_require_nifty_()->get_all<T_>(key_, including_empty_);	\
	}	\
	//
#define DEFINE_MODULE_CONFIG(path_)	\
	/* Define the config file for this module. */	\
	namespace {	\
		::boost::weak_ptr<const ::Poseidon::ConfigFile> g_weak_module_config_nifty_;	\
	}	\
	/* Define the initialization callback. It must have a very high priority. */	\
	MODULE_RAII_PRIORITY(handles_, LONG_MIN){	\
		const AUTO(module_config_, ::boost::make_shared< ::Poseidon::ConfigFile>(path_));	\
		handles_.push(module_config_);	\
		g_weak_module_config_nifty_ = module_config_;	\
	}	\
	/* Define the access function for this module. This function will not return a null pointer. */	\
	::boost::shared_ptr<const ::Poseidon::ConfigFile> module_config_require_nifty_(){	\
		AUTO(module_config_, g_weak_module_config_nifty_.lock());	\
		DEBUG_THROW_UNLESS(module_config_, ::Poseidon::Exception, ::Poseidon::sslit("The configuration file for this module has not been loaded"));	\
		return module_config_;	\
	}	\
	//

#endif
