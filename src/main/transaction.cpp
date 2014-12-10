// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "transaction.hpp"
using namespace Poseidon;

TransactionItemBase::~TransactionItemBase(){
}

bool Transaction::empty() const {
	return m_items.empty();
}
void Transaction::add(boost::shared_ptr<TransactionItemBase> item){
	m_items.push_back(STD_MOVE(item));
}
void Transaction::clear(){
	m_items.clear();
}

bool Transaction::commit() const {
	AUTO(it, m_items.begin());
	try {
		while(it != m_items.end()){
			if(!(*it)->lock()){
				break;
			}
			++it;
		}
	} catch(...){
		while(it != m_items.begin()){
			--it;
			(*it)->unlock();
		}
		throw;
	}
	if(it != m_items.end()){
		while(it != m_items.begin()){
			--it;
			(*it)->unlock();
		}
		return false;
	}
	it = m_items.begin();
	while(it != m_items.end()){
		(*it)->commit();
		++it;
	}
	while(it != m_items.begin()){
		--it;
		(*it)->unlock();
	}
	return true;
}
