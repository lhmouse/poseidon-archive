#ifndef POSEIDON_TRANSACTION_HPP_
#define POSEIDON_TRANSACTION_HPP_

#include "../cxx_ver.hpp"
#include <vector>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/noncopyable.hpp>

namespace Poseidon {

class TransactionItemBase : boost::noncopyable {
	friend class Transaction;

private:
	virtual bool lock() = 0;
	virtual void unlock() throw() = 0;
	virtual void commit() throw() = 0;
};

template<class LockT, class UnlockT, class CommitT>
class TransactionItem : public TransactionItemBase {
private:
	const LockT m_lock;
	const UnlockT m_unlock;
	const CommitT m_commit;

public:
	TransactionItem(LockT lock, UnlockT unlock, CommitT commit)
		: m_lock(STD_MOVE(lock)), m_unlock(STD_MOVE(unlock)), m_commit(STD_MOVE(commit))
	{
	}

private:
	virtual bool lock(){
		return m_lock();
	}
	virtual void unlock() throw() {
		try {
			m_unlock();
		} catch(...){
		}
	}
	virtual void commit() throw() {
		try {
			m_commit();
		} catch(...){
		}
	}
};

class Transaction : boost::noncopyable {
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
