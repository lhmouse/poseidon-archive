// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_HEADER_OPTION_HPP_
#define POSEIDON_HTTP_HEADER_OPTION_HPP_

#include "../optional_map.hpp"

namespace Poseidon {

namespace Http {
	class HeaderOption {
	private:
		std::string m_base;
		OptionalMap m_options;

	public:
		explicit HeaderOption(const std::string &str);
		HeaderOption(std::string base, OptionalMap options);

	public:
		const std::string &get_base() const {
			return m_base;
		}
		std::string &get_base(){
			return m_base;
		}
		void set_base(std::string base){
			m_base = STD_MOVE(base);
		}
		const OptionalMap &get_options() const {
			return m_options;
		}
		OptionalMap &get_options(){
			return m_options;
		}
		void set_options(OptionalMap options){
			m_options = STD_MOVE(options);
		}

		const std::string &get_option(const char *key) const {
			return m_options.get(key);
		}
		const std::string &get_option(const SharedNts &key) const {
			return m_options.get(key);
		}
		void set_option(SharedNts key, std::string value){
			m_options.set(STD_MOVE(key), STD_MOVE(value));
		}
		bool erase_option(const char *key){
			return m_options.erase(key);
		}
		bool erase_option(const SharedNts &key){
			return m_options.erase(key);
		}

		std::string to_string() const;
	};

	inline std::ostream &operator<<(std::ostream &os, const HeaderOption &rhs){
		return os << rhs.to_string();
	}
}

}

#endif
