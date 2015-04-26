// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_TRANSACTION_HPP_
#define POSEIDON_TRANSACTION_HPP_

#include "cxx_util.hpp"
#include <vector>
#include <exception>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

namespace Poseidon {

class TransactionItemBase : NONCOPYABLE {
protected:
	static void logIgnoredStdException(const char *what) NOEXCEPT;
	static void logIgnoredUnknownException() NOEXCEPT;

public:
	virtual ~TransactionItemBase();

public:
	virtual bool lock() = 0;
	virtual void unlock() NOEXCEPT = 0;
	virtual void commit() NOEXCEPT = 0;
};

template<class LockT, class UnlockT, class CommitT>
class TransactionItem FINAL : public TransactionItemBase {
private:
	const LockT m_lock;
	const UnlockT m_unlock;
	const CommitT m_commit;

public:
	TransactionItem(LockT lock, UnlockT unlock, CommitT commit)
		: m_lock(STD_MOVE(lock)), m_unlock(STD_MOVE(unlock)), m_commit(STD_MOVE(commit))
	{
	}

public:
	virtual bool lock() OVERRIDE {
		return m_lock();
	}
	virtual void unlock() NOEXCEPT OVERRIDE {
		try {
			m_unlock();
		} catch(std::exception &e){
			logIgnoredStdException(e.what());
		} catch(...){
			logIgnoredUnknownException();
		}
	}
	virtual void commit() NOEXCEPT OVERRIDE {
		try {
			m_commit();
		} catch(std::exception &e){
			logIgnoredStdException(e.what());
		} catch(...){
			logIgnoredUnknownException();
		}
	}
};

class Transaction : NONCOPYABLE {
private:
	std::vector<boost::shared_ptr<TransactionItemBase> > m_items;

public:
	bool empty() const;
	void add(boost::shared_ptr<TransactionItemBase> item);
	void clear();

	bool commit() const;

	template<class LockT, class UnlockT, class CommitT>
	void add(LockT lock, UnlockT unlock, CommitT commit){
		add(boost::make_shared<TransactionItem<LockT, UnlockT, CommitT> >(
			STD_MOVE(lock), STD_MOVE(unlock), STD_MOVE(commit)));
	}
};

}

#endif
