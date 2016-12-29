#include <poseidon/precompiled.hpp>
#include <poseidon/log.hpp>
#include <poseidon/async_job.hpp>
#include <poseidon/module_raii.hpp>
#include <poseidon/singletons/job_dispatcher.hpp>
#include <poseidon/singletons/mongodb_daemon.hpp>
#include <poseidon/mongodb/object_base.hpp>

namespace {

#define MONGODB_OBJECT_NAME         Foo
#define MONGODB_OBJECT_PRIMARY_KEY  { return m_string; }
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
			auto promise = Poseidon::MongoDbDaemon::enqueue_for_batch_loading(
				[=](const boost::shared_ptr<Poseidon::MongoDb::Connection> &conn){
					auto obj = boost::make_shared<Foo>();
					obj->fetch(conn);
					LOG_POSEIDON_FATAL("Loaded: boolean = ", obj->m_boolean, ", signed = ", obj->m_signed,
						", unsigned = ", obj->m_unsigned, ", double = ", obj->m_double, ", string = ", obj->m_string,
						", datetime = ", obj->m_datetime, ", uuid = ", obj->m_uuid, ", blob = ", obj->m_blob);
				}, "Foo", { }, 0, 10);
			Poseidon::JobDispatcher::yield(promise, true);

			auto foo = boost::make_shared<Foo>(true, -123, 456, 78.9, "hello world!", UINT64_MAX,
				Poseidon::Uuid("986CDBFA-E18E-F8C6-A1D2-4CB13B1100D5"), "binary");
			foo->async_save(true, true);
			LOG_POSEIDON_FATAL("Saved!");
		});
}
