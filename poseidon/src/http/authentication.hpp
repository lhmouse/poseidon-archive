// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_AUTHENTICATION_HPP_
#define POSEIDON_HTTP_AUTHENTICATION_HPP_

#include <string>
#include <boost/shared_ptr.hpp>
#include "request_headers.hpp"
#include "../ip_port.hpp"

namespace Poseidon {
namespace Http {

enum Authentication_result {
	auth_succeeded               = 0,
	auth_header_not_set          = 1,
	auth_header_format_error     = 2,
	auth_scheme_not_supported    = 3,
	auth_password_incorrect      = 4,
	auth_request_expired         = 5,
	auth_algorithm_not_supported = 7,
	auth_qop_not_supported       = 8,
};

class Authentication_context; // 没有定义的类，当作句柄使用。

// 以下是通用接口。
// 创建一个认证上下文，参数 basic_user_pass 应当包含一系列的 username:password 字符串且不得为空。
extern boost::shared_ptr<const Authentication_context> create_authentication_context(
	const std::string &realm, const boost::container::vector<std::string> &basic_user_pass);
// 支持 Basic 和 Digest 方式认证，如果参数 auth_info 为空返回成功。
extern std::pair<Authentication_result, const char *> check_authentication(
	const boost::shared_ptr<const Authentication_context> &context, bool is_proxy, const Ip_port &remote_info, const Request_headers &request_headers);
// 建议 Digest 方式认证。
__attribute__((__noreturn__)) extern void throw_authentication_failure(
	const boost::shared_ptr<const Authentication_context> &context, bool is_proxy, const Ip_port &remote_info, Authentication_result result);
// 一站式接口：如果调用 check_authentication() 并成功则正常返回，否则调用 throw_authentication_failure() 而不返回。
extern const char *check_authentication_simple(
	const boost::shared_ptr<const Authentication_context> &context, bool is_proxy, const Ip_port &remote_info, const Request_headers &request_headers);

// 以下是各个认证方式的独立接口。
// Basic
extern std::pair<Authentication_result, const char *> check_authentication_basic(
	const boost::shared_ptr<const Authentication_context> &context, const std::string &header_value);
__attribute__((__noreturn__)) extern void throw_authentication_failure_basic(
	const boost::shared_ptr<const Authentication_context> &context, bool is_proxy, Authentication_result result);
// Digest
extern std::pair<Authentication_result, const char *> check_authentication_digest(
	const boost::shared_ptr<const Authentication_context> &context, const Ip_port &remote_info, Verb verb, const std::string &header_value);
__attribute__((__noreturn__)) extern void throw_authentication_failure_digest(
	const boost::shared_ptr<const Authentication_context> &context, bool is_proxy, const Ip_port &remote_info, Authentication_result result);

}
}

#endif
