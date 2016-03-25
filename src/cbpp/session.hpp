// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CBPP_SESSION_HPP_
#define POSEIDON_CBPP_SESSION_HPP_

#include "low_level_session.hpp"

namespace Poseidon {

namespace Cbpp {
	class Session : public LowLevelSession {
	private:
		class SyncJobBase;
		class DataMessageJob;
		class ControlMessageJob;

	private:
		const boost::uint64_t m_max_request_length;

		boost::uint64_t m_size_total;
		unsigned m_message_id;
		StreamBuffer m_payload;

	public:
		explicit Session(UniqueFile socket, boost::uint64_t max_request_length = 0);
		~Session();

	protected:
		// LowLevelSession
		void on_read_avail(StreamBuffer data) OVERRIDE;

		void on_low_level_data_message_header(boost::uint16_t message_id, boost::uint64_t payload_size) OVERRIDE;
		void on_low_level_data_message_payload(boost::uint64_t payload_offset, StreamBuffer payload) OVERRIDE;
		bool on_low_level_data_message_end(boost::uint64_t payload_size) OVERRIDE;

		bool on_low_level_control_message(ControlCode control_code, boost::int64_t vint_param, std::string string_param) OVERRIDE;

		// 可覆写。
		virtual void on_sync_data_message(boost::uint16_t message_id, StreamBuffer payload) = 0;
		virtual void on_sync_control_message(ControlCode control_code, boost::int64_t vint_param, std::string string_param);
	};
}

}

#endif
