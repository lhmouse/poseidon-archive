#include "../precompiled.hpp"
#include "client.hpp"
#include "exception.hpp"
#include "status_codes.hpp"
#include "../log.hpp"
#include "../job_base.hpp"
#include "../profiler.hpp"

namespace Poseidon {

namespace Http {
	namespace {
		class ClientJobBase : public JobBase {
		private:
			const boost::weak_ptr<Client> m_client;

		protected:
			explicit ClientJobBase(const boost::shared_ptr<Client> &client)
				: m_client(client)
			{
			}

		protected:
			virtual void perform(const boost::shared_ptr<Client> &client) const = 0;

		private:
			boost::weak_ptr<const void> getCategory() const FINAL {
				return m_client;
			}
			void perform() const FINAL {
				PROFILE_ME;

				const AUTO(client, m_client.lock());
				if(!client){
					return;
				}

				try {
					perform(client);
				} catch(TryAgainLater &){
					throw;
				} catch(std::exception &e){
					LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "std::exception thrown: what = ", e.what());
					client->forceShutdown();
					throw;
				} catch(...){
					LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Unknown exception thrown.");
					client->forceShutdown();
					throw;
				}
			}
		};
	}

	class Client::ResponseHeadersJob : public ClientJobBase {
	private:
		const ResponseHeaders m_responseHeaders;
		const std::string m_transferEncoding;
		const boost::uint64_t m_contentLength;

	public:
		ResponseHeadersJob(const boost::shared_ptr<Client> &client,
			ResponseHeaders responseHeaders, std::string transferEncoding, boost::uint64_t contentLength)
			: ClientJobBase(client)
			, m_responseHeaders(STD_MOVE(responseHeaders)), m_transferEncoding(STD_MOVE(transferEncoding)), m_contentLength(contentLength)
		{
		}

	protected:
		void perform(const boost::shared_ptr<Client> &client) const OVERRIDE {
			PROFILE_ME;

			client->onSyncResponseHeaders(m_responseHeaders, m_transferEncoding, m_contentLength);
		}
	};

	class Client::ResponseEntityJob : public ClientJobBase {
	private:
		const boost::uint64_t m_contentOffset;
		const StreamBuffer m_entity;

	public:
		ResponseEntityJob(const boost::shared_ptr<Client> &client, boost::uint64_t contentOffset, StreamBuffer entity)
			: ClientJobBase(client)
			, m_contentOffset(contentOffset), m_entity(STD_MOVE(entity))
		{
		}

	protected:
		void perform(const boost::shared_ptr<Client> &client) const OVERRIDE {
			PROFILE_ME;

			client->onSyncResponseEntity(m_contentOffset, m_entity);
		}
	};

	class Client::ResponseEndJob : public ClientJobBase {
	private:
		const boost::uint64_t m_contentLength;
		const bool m_isChunked;
		const OptionalMap m_headers;

	public:
		ResponseEndJob(const boost::shared_ptr<Client> &client, boost::uint64_t contentLength, bool isChunked, OptionalMap headers)
			: ClientJobBase(client)
			, m_contentLength(contentLength), m_isChunked(isChunked), m_headers(STD_MOVE(headers))
		{
		}

	protected:
		void perform(const boost::shared_ptr<Client> &client) const OVERRIDE {
			PROFILE_ME;

			client->onSyncResponseEnd(m_contentLength, m_isChunked, m_headers);
		}
	};

	Client::Client(const SockAddr &addr, bool useSsl)
		: TcpClientBase(addr, useSsl)
	{
	}
	Client::Client(const IpPort &addr, bool useSsl)
		: TcpClientBase(addr, useSsl)
	{
	}
	Client::~Client(){
	}

	void Client::onReadHup() NOEXCEPT {
		PROFILE_ME;

		try {
			if(ClientReader::isContentTillEof()){
				terminateContent();
			}
		} catch(std::exception &e){
			LOG_POSEIDON_WARNING("std::exception thrown: what = ", e.what());
			forceShutdown();
		} catch(...){
			LOG_POSEIDON_WARNING("Unknown exception thrown");
			forceShutdown();
		}

		TcpSessionBase::onReadHup();
	}

	void Client::onReadAvail(const void *data, std::size_t size){
		PROFILE_ME;

		ClientReader::putEncodedData(StreamBuffer(data, size));
	}
	void Client::onResponseHeaders(ResponseHeaders responseHeaders, std::string transferEncoding, boost::uint64_t contentLength){
		PROFILE_ME;

		enqueueJob(boost::make_shared<ResponseHeadersJob>(
			virtualSharedFromThis<Client>(), STD_MOVE(responseHeaders), STD_MOVE(transferEncoding), contentLength));
	}
	void Client::onResponseEntity(boost::uint64_t entityOffset, StreamBuffer entity){
		PROFILE_ME;

		enqueueJob(boost::make_shared<ResponseEntityJob>(
			virtualSharedFromThis<Client>(), entityOffset, STD_MOVE(entity)));
	}
	bool Client::onResponseEnd(boost::uint64_t contentLength, bool isChunked, OptionalMap headers){
		PROFILE_ME;

		enqueueJob(boost::make_shared<ResponseEndJob>(
			virtualSharedFromThis<Client>(), contentLength, isChunked, STD_MOVE(headers)));

		return true;
	}

	long Client::onEncodedDataAvail(StreamBuffer encoded){
		PROFILE_ME;

		return TcpSessionBase::send(STD_MOVE(encoded));
	}
}

}
