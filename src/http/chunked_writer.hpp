// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_CHUNKED_WRITER_HPP_
#define POSEIDON_HTTP_CHUNKED_WRITER_HPP_

#include "../cxx_util.hpp"
#include <boost/shared_ptr.hpp>
#include "status_codes.hpp"
#include "../optional_map.hpp"
#include "../stream_buffer.hpp"

namespace Poseidon {

namespace Http {
	class Session;

	class ChunkedWriter : NONCOPYABLE {
	private:
		boost::shared_ptr<Session> m_session;

	public:
		ChunkedWriter();
		ChunkedWriter(boost::shared_ptr<Session> session, StatusCode statusCode, OptionalMap headers = OptionalMap());
		// 在析构之前必须手动调用 finalize() 或 reset()。
		// 如果有异常抛出而导致析构，连接会被非正常关闭。
		~ChunkedWriter();

	public:
		const boost::shared_ptr<Session> &getSession() const {
			return m_session;
		}

		void reset() NOEXCEPT;
		void reset(boost::shared_ptr<Session> session, StatusCode statusCode, OptionalMap headers = OptionalMap());

		void put(const void *data, std::size_t size){
			put(StreamBuffer(data, size));
		}
		void put(const char *str){
			put(StreamBuffer(str));
		}
		void put(StreamBuffer buffer);

		void finalize(OptionalMap headers = OptionalMap());
	};
}

}

#endif
