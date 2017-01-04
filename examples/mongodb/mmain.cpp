#include <poseidon/precompiled.hpp>
#include <poseidon/log.hpp>
#include <poseidon/async_job.hpp>
#include <poseidon/module_raii.hpp>
#include <poseidon/singletons/job_dispatcher.hpp>
#include <poseidon/singletons/mongodb_daemon.hpp>
#include <poseidon/mongodb/object_base.hpp>

namespace {

#define MONGODB_OBJECT_NAME         foo
#define MONGODB_OBJECT_PRIMARY_KEY  { return m_uuid.unlocked_get().to_string(); }
#define MONGODB_OBJECT_FIELDS \
	FIELD_BOOLEAN             (m_boolean)	\
	FIELD_SIGNED              (m_signed)	\
	FIELD_UNSIGNED            (m_unsigned)	\
	FIELD_DOUBLE              (m_double)	\
	FIELD_STRING              (m_string)	\
	FIELD_DATETIME            (m_datetime)	\
	FIELD_UUID                (m_uuid)	\
	FIELD_BLOB                (m_blob)
#include <poseidon/mongodb/object_generator.hpp>

}

MODULE_RAII(){
	Poseidon::enqueue_async_job(
		{ },
		[]{
			Poseidon::MongoDb::BsonBuilder query;
			query.append_string(Poseidon::sslit("find"), "foo");
			query.append_object(Poseidon::sslit("filter"), { });
			query.append_unsigned(Poseidon::sslit("batchSize"), 2);
			auto promise = Poseidon::MongoDbDaemon::enqueue_for_batch_loading(
				[=](const boost::shared_ptr<Poseidon::MongoDb::Connection> &conn){
					auto obj = boost::make_shared<foo>();
					obj->fetch(conn);
					LOG_POSEIDON_FATAL("Loaded: boolean = ", obj->m_boolean, ", signed = ", obj->m_signed,
						", unsigned = ", obj->m_unsigned, ", double = ", obj->m_double, ", string = ", obj->m_string,
						", datetime = ", obj->m_datetime, ", uuid = ", obj->m_uuid, ", blob = ", obj->m_blob);
				}, "foo", std::move(query));
			Poseidon::JobDispatcher::yield(promise, true);

			auto f = boost::make_shared<foo>(true, -123, 456, 78.9, "hello world!", UINT64_MAX, Poseidon::Uuid::random(), "binary");
			f->async_save(true, true);
			LOG_POSEIDON_FATAL("Saved!");
		});
}
