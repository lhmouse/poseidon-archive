// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "basic_auth_server.hpp"

namespace Poseidon {

namespace Http {
	namespace {
		boost::shared_ptr<std::vector<std::string> > createAuthInfo(std::vector<std::string> authInfo){
			if(authInfo.empty()){
				return VAL_INIT;
			}
			std::sort(authInfo.begin(), authInfo.end());
			return boost::make_shared<std::vector<std::string> >(STD_MOVE(authInfo));
		}
		std::string createPath(std::string str){
			if(str.begin()[0] != '/'){
				str.insert(str.begin(), '/');
			}
			if(str.end()[-1] == '/'){
				str.erase(str.end() - 1);
			}
			return STD_MOVE(str);
		}
	}

	BasicAuthServer::BasicAuthServer(const IpPort &bindAddr, const char *cert, const char *privateKey,
		std::vector<std::string> authInfo, std::string path)
		: TcpServerBase(bindAddr, cert, privateKey)
		, m_authInfo(createAuthInfo(STD_MOVE(authInfo))), m_path(createPath(STD_MOVE(path)))
	{
	}
	BasicAuthServer::~BasicAuthServer(){
	}
}

}
