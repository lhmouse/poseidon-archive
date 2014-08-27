#ifndef POSEIDON_TRANSACTION_HPP_
#define POSEIDON_TRANSACTION_HPP_

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

template<class Lock, class Unlock, class Commit>
class TransactionItem : public TransactionItemBase {
private:
    const Lock m_lock;
    const Unlock m_unlock;
    const Commit m_commit;

public:
    TransactionItem(Lock lock, Unlock unlock, Commit commit)
        : m_lock(lock), m_unlock(unlock), m_commit(commit)
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

    template<class Lock, class Unlock, class Commit>
    void add(Lock lock, Unlock unlock, Commit commit){
        add(boost::make_shared<TransactionItem<Lock, Unlock, Commit> >(lock, unlock, commit));
    }
};

}

#endif
