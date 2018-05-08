// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_HEADER_OPTION_HPP_
#define POSEIDON_HTTP_HEADER_OPTION_HPP_

#include "../optional_map.hpp"
#include "../fwd.hpp"
#include <iosfwd>

namespace Poseidon {
namespace Http {

class Header_option {
private:
	std::string m_base;
	Optional_map m_options;

public:
	Header_option()
		: m_base(), m_options()
	{
		//
	}
	Header_option(std::string base, Optional_map options = Optional_map())
		: m_base(STD_MOVE(base)), m_options(STD_MOVE(options))
	{
		//
	}
	explicit Header_option(std::istream &is);

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
	const Optional_map &get_options() const {
		return m_options;
	}
	Optional_map &get_options(){
		return m_options;
	}
	void set_options(Optional_map options){
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
	const std::string &get_option(const Rcnts &key) const {
		return m_options.get(key);
	}
	void set_option(Rcnts key, std::string value){
		m_options.set(STD_MOVE(key), STD_MOVE(value));
	}
	bool erase_option(const char *key){
		return m_options.erase(key);
	}
	bool erase_option(const Rcnts &key){
		return m_options.erase(key);
	}

	void swap(Header_option &rhs) NOEXCEPT {
		using std::swap;
		swap(m_base, rhs.m_base);
		swap(m_options, rhs.m_options);
	}

	Stream_buffer dump() const;
	void dump(std::ostream &os) const;
	void parse(std::istream &is);
};

inline void swap(Header_option &lhs, Header_option &rhs) NOEXCEPT {
	lhs.swap(rhs);
}

inline std::ostream &operator<<(std::ostream &os, const Header_option &rhs){
	rhs.dump(os);
	return os;
}
inline std::istream &operator>>(std::istream &is, Header_option &rhs){
	rhs.parse(is);
	return is;
}

}
}

#endif
