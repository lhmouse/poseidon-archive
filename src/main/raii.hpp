#ifndef POSEIDON_RAII_HPP_
#define POSEIDON_RAII_HPP_

namespace Poseidon {

template<typename Handle, typename Closer>
class ScopedHandle {
private:
	Handle m_handle;

public:
	ScopedHandle() throw()
		: m_handle(Closer()())
	{
	}
	explicit ScopedHandle(Handle handle)
		: m_handle(Closer()())
	{
		reset(handle);
	}
	ScopedHandle &operator=(Handle handle) throw() {
		reset(handle);
		return *this;
	}
	~ScopedHandle() throw() {
		reset();
	}

private:
	ScopedHandle(const ScopedHandle &);
	void operator=(const ScopedHandle &);

public:
	Handle get() const throw() {
		return m_handle;
	}
	Handle release() throw() {
		const Handle ret = m_handle;
		m_handle = Closer()();
		return ret;
	}
	void reset(Handle handle = Closer()()) throw() {
		const Handle old = m_handle;
		m_handle = handle;
		if(old != Closer()()){
			Closer()(old);
		}
	}
	void reset(ScopedHandle &rhs) throw() {
		if(&rhs == this){
			return;
		}
		swap(rhs);
		rhs.reset();
	}
	void swap(ScopedHandle &rhs) throw() {
		const Handle my = m_handle;
		m_handle = rhs.m_handle;
		rhs.m_handle = my;
	}
	operator bool() const throw() {
		return get() != Closer()();
	}
};

template<typename Handle, typename Closer>
class SharedHandle {
private:
	typedef ScopedHandle<Handle, Closer> Scoped;

	struct Control {
		Scoped h;
		volatile unsigned long ref;
	};

private:
	Control *m_control;

public:
	SharedHandle() throw()
		: m_control(0)
	{
	}
	explicit SharedHandle(Handle handle)
		: m_control(0)
	{
		reset(handle);
	}
	SharedHandle(const SharedHandle &rhs) throw()
		: m_control(0)
	{
		reset(rhs);
	}
	SharedHandle &operator=(Handle handle){
		Scoped tmp(handle);
		reset(tmp);
		return *this;
	}
	SharedHandle &operator=(const SharedHandle &rhs) throw() {
		reset(rhs);
		return *this;
	}
	~SharedHandle() throw() {
		reset();
	}

public:
	Handle get() const throw() {
		if(!m_control){
			return Closer()();
		}
		return m_control->h.get();
	}
	bool unique() const throw() {
		if(!m_control){
			return false;
		}
		volatile int barrier;
		__sync_lock_test_and_set(&barrier, 1);
		return m_control->ref == 1;
	}
	void reset() throw() {
		if(!m_control){
			return;
		}
		if(__sync_sub_and_fetch(&m_control->ref, 1) == 0){
			delete m_control;
		}
		m_control = 0;
	}
	void reset(Handle handle){
		Scoped tmp(handle);
		reset(tmp);
	}
	void reset(Scoped &scoped){
		if(!scoped){
			reset();
			return;
		}
		if(unique()){
			m_control->h.swap(scoped);
			scoped.reset();
		} else {
			reset();
			m_control = new Control;
			m_control->h.swap(scoped);
			m_control->ref = 1;
			volatile int barrier;
			__sync_lock_release(&barrier);
		}
	}
	void reset(const SharedHandle &rhs) throw() {
		if(&rhs == this){
			return;
		}
		if(rhs.m_control){
			m_control = rhs.m_control;
			__sync_add_and_fetch(&m_control->ref, 1);
		} else {
			reset();
		}
	}
	void swap(SharedHandle &rhs) throw() {
		Control *const my = m_control;
		m_control = rhs.m_control;
		rhs.m_control = my;
	}
	operator bool() const throw() {
		return get() != Closer()();
	}
};

template<typename Handle, typename Closer>
void swap(ScopedHandle<Handle, Closer> &lhs,
	ScopedHandle<Handle, Closer> &rhs) throw()
{
	lhs.swap(rhs);
}

template<typename Handle, typename Closer>
void swap(SharedHandle<Handle, Closer> &lhs,
	SharedHandle<Handle, Closer> &rhs) throw()
{
	lhs.swap(rhs);
}

#define DEFINE_RATIONAL_OPERATOR_(temp_, op_)	\
	template<typename Handle, typename Closer>	\
	bool operator op_(const temp_<Handle, Closer> &lhs,	\
		const temp_<Handle, Closer> &rhs) throw()	\
	{	\
		return lhs.get() op_ rhs.get();	\
	}	\
	template<typename Handle, typename Closer>	\
	bool operator op_(Handle lhs,	\
		const temp_<Handle, Closer> &rhs) throw()	\
	{	\
		return lhs op_ rhs.get();	\
	}	\
	template<typename Handle, typename Closer>	\
	bool operator op_(const temp_<Handle, Closer> &lhs,	\
		Handle rhs) throw()	\
	{	\
		return lhs.get() op_ rhs;	\
	}

DEFINE_RATIONAL_OPERATOR_(ScopedHandle, ==)
DEFINE_RATIONAL_OPERATOR_(ScopedHandle, !=)
DEFINE_RATIONAL_OPERATOR_(ScopedHandle, <)
DEFINE_RATIONAL_OPERATOR_(ScopedHandle, >)
DEFINE_RATIONAL_OPERATOR_(ScopedHandle, <=)
DEFINE_RATIONAL_OPERATOR_(ScopedHandle, >=)

DEFINE_RATIONAL_OPERATOR_(SharedHandle, ==)
DEFINE_RATIONAL_OPERATOR_(SharedHandle, !=)
DEFINE_RATIONAL_OPERATOR_(SharedHandle, <)
DEFINE_RATIONAL_OPERATOR_(SharedHandle, >)
DEFINE_RATIONAL_OPERATOR_(SharedHandle, <=)
DEFINE_RATIONAL_OPERATOR_(SharedHandle, >=)

#undef DEFINE_RATIONAL_OPERATOR_

extern void closeFile(int fd) throw();

struct FileCloser {
	int operator()() const throw() {
		return -1;
	}
	void operator()(int fd) const throw() {
		closeFile(fd);
	}
};

typedef ScopedHandle<int, FileCloser> ScopedFile;
typedef SharedHandle<int, FileCloser> SharedFile;

}

#endif
