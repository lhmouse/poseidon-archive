#include "../precompiled.hpp"
#include "transaction.hpp"
using namespace Poseidon;

bool Transaction::empty() const {
	return m_items.empty();
}
void Transaction::add(boost::shared_ptr<TransactionItemBase> item){
	m_items.push_back(item);
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
