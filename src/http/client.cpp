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

	class Client::ResponseHeaderJob : public ClientJobBase {
	private:
		const ResponseHeaders m_responseHeaders;
		const std::vector<std::string> m_transferEncoding;
		const boost::uint64_t m_contentLength;

	public:
		ResponseHeaderJob(const boost::shared_ptr<Client> &client, ResponseHeaders responseHeaders,
			std::vector<std::string> m_transferEncoding, boost::uint64_t contentLength)
			: ClientJobBase(client)
			, m_responseHeaders(STD_MOVE(responseHeaders))
			, m_transferEncoding(STD_MOVE(m_transferEncoding)), m_contentLength(contentLength)
		{
		}

	protected:
		void perform(const boost::shared_ptr<Client> &client) const OVERRIDE {
			PROFILE_ME;
			LOG_POSEIDON_DEBUG("Dispatching response header: statusCode = ", m_responseHeaders.statusCode);

			client->onResponseHeaders(m_responseHeaders, m_transferEncoding, m_contentLength);
		}
	};

	class Client::EntityJob : public ClientJobBase {
	private:
		const boost::uint64_t m_contentOffset;
		const StreamBuffer m_entity;

	public:
		EntityJob(const boost::shared_ptr<Client> &client, boost::uint64_t contentOffset, StreamBuffer entity)
			: ClientJobBase(client)
			, m_contentOffset(contentOffset), m_entity(STD_MOVE(entity))
		{
		}

	protected:
		void perform(const boost::shared_ptr<Client> &client) const OVERRIDE {
			PROFILE_ME;
			LOG_POSEIDON_DEBUG("Dispatching response entity: contentOffset = ", m_contentOffset, ", size = ", m_entity.size());

			client->onEntity(m_contentOffset, m_entity);
		}
	};

	class Client::ResponseEofJob : public ClientJobBase {
	private:
		const boost::uint64_t m_realContentLength;
		const OptionalMap m_headers;

	public:
		ResponseEofJob(const boost::shared_ptr<Client> &client, boost::uint64_t realContentLength, OptionalMap headers)
			: ClientJobBase(client)
			, m_realContentLength(realContentLength), m_headers(STD_MOVE(headers))
		{
		}

	protected:
		void perform(const boost::shared_ptr<Client> &client) const OVERRIDE {
			PROFILE_ME;
			LOG_POSEIDON_DEBUG("Dispatching chunked trailer");

			client->onResponseEof(m_realContentLength, m_headers);
		}
	};

	Client::Client(const SockAddr &addr, bool useSsl)
		: LowLevelClient(addr, useSsl)
	{
	}
	Client::Client(const IpPort &addr, bool useSsl)
		: LowLevelClient(addr, useSsl)
	{
	}
	Client::~Client(){
	}

	void Client::onLowLevelResponseHeaders(ResponseHeaders responseHeaders,
		std::vector<std::string> transferEncoding, boost::uint64_t contentLength)
	{
		PROFILE_ME;

		enqueueJob(boost::make_shared<ResponseHeaderJob>(
			virtualSharedFromThis<Client>(), STD_MOVE(responseHeaders), STD_MOVE(transferEncoding), contentLength));
	}
	void Client::onLowLevelEntity(boost::uint64_t contentOffset, StreamBuffer entity){
		PROFILE_ME;

		enqueueJob(boost::make_shared<EntityJob>(
			virtualSharedFromThis<Client>(), contentOffset, STD_MOVE(entity)));
	}
	void Client::onLowLevelResponseEof(boost::uint64_t realContentLength, OptionalMap headers){
		PROFILE_ME;

		enqueueJob(boost::make_shared<ResponseEofJob>(
			virtualSharedFromThis<Client>(), realContentLength, STD_MOVE(headers)));
	}
}

}
