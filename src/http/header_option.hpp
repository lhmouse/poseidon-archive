// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_HEADER_OPTION_HPP_
#define POSEIDON_HTTP_HEADER_OPTION_HPP_

#include <iosfwd>
#include "../optional_map.hpp"

namespace Poseidon {

namespace Http {
	class HeaderOption {
	private:
		std::string m_base;
		OptionalMap m_options;

	public:
		HeaderOption()
			: m_base(), m_options()
		{
		}
		HeaderOption(std::string base, OptionalMap options)
			: m_base(STD_MOVE(base)), m_options(STD_MOVE(options))
		{
		}
		explicit HeaderOption(std::istream &is)
			: m_base(), m_options()
		{
			parse(is);
		}

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

		bool empty() const {
			return m_base.empty();
		}
		void clear(){
			m_base.clear();
			m_options.clear();
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

		void swap(HeaderOption &rhs) NOEXCEPT {
			using std::swap;
			swap(m_base, rhs.m_base);
			swap(m_options, rhs.m_options);
		}

		std::string dump() const;
		void dump(std::ostream &os) const;
		void parse(std::istream &is);
	};

	inline void swap(HeaderOption &lhs, HeaderOption &rhs) NOEXCEPT {
		lhs.swap(rhs);
	}

	inline std::ostream &operator<<(std::ostream &os, const HeaderOption &rhs){
		rhs.dump(os);
		return os;
	}
	inline std::istream &operator>>(std::istream &is, HeaderOption &rhs){
		rhs.parse(is);
		return is;
	}
}

}

#endif
