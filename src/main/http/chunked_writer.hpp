#ifndef POSEIDON_HTTP_CHUNKED_WRITER_HPP_
#define POSEIDON_HTTP_CHUNKED_WRITER_HPP_

#include "../cxx_util.hpp"
#include <boost/shared_ptr.hpp>
#include "status_codes.hpp"

namespace Poseidon {

class OptionalMap;
class StreamBuffer;

namespace Http {
	class Session;

	class ChunkedWriter : NONCOPYABLE {
	private:
		boost::shared_ptr<Session> m_session;
		bool m_inProgress;

	public:
		explicit ChunkedWriter(boost::shared_ptr<Session> session = VAL_INIT);
		// 在析构之前必须手动调用 finalize() 或 reset()。
		// 如果有异常抛出而导致析构，连接会被非正常关闭。
		~ChunkedWriter();

	public:
		const boost::shared_ptr<Session> &getSession() const {
			return m_session;
		}
		bool isInProgress() const {
			return m_inProgress;
		}

		void reset(boost::shared_ptr<Session> session = VAL_INIT) NOEXCEPT;

		void initialize(StatusCode statusCode, OptionalMap headers = VAL_INIT);

		void put(const void *data, std::size_t size);
		void put(const char *str);
		void put(StreamBuffer buffer);

		void finalize(OptionalMap headers, bool fin = false);
		void finalize(bool fin = false);
	};
}

}

#endif
