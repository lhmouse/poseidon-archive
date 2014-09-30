#include "../precompiled.hpp"
#include <boost/bind.hpp>
#include "../main/module.hpp"
#include "../main/log.hpp"
#include "../main/exception.hpp"
#include "../main/singletons/http_servlet_manager.hpp"
#include "../main/singletons/module_manager.hpp"
using namespace Poseidon;

namespace {

boost::shared_ptr<const HttpServlet> g_load, g_unload, g_meow;

HttpStatus meowProc(OptionalMap &headers, StreamBuffer &contents, HttpRequest){
	headers.set("Content-Type", "text/html");
	contents.put("<h1>Meow!</h1>");
	return HTTP_OK;
}
HttpStatus loadProc(OptionalMap &, StreamBuffer &contents, HttpRequest,
	const boost::weak_ptr<const Module> &module)
{
	if(g_meow){
		contents.put("Already loaded");
		return HTTP_OK;
	}
	// 通配路径 /meow/*
	g_meow = HttpServletManager::registerServlet("/meow/", module, &meowProc);
	contents.put("OK");
	return HTTP_OK;
}
HttpStatus unloadProc(OptionalMap &, StreamBuffer &contents, HttpRequest request){
	if(request.getParams["unload_module"] == "1"){
		ModuleManager::unload("libposeidon-template.so");
		contents.put("Module unloaded");
		return HTTP_OK;
	}
	if(!g_meow){
		contents.put("Already unloaded");
		return HTTP_OK;
	}
	g_meow.reset();
	contents.put("OK");
	return HTTP_OK;
}

}

extern "C" void poseidonModuleInit(const boost::weak_ptr<const Module> &module){
	LOG_FATAL("poseidonModuleInit()");

	g_load = HttpServletManager::registerServlet("/load", module,
		boost::bind(&loadProc, _1, _2, _3, module));
	g_unload = HttpServletManager::registerServlet("/unload", module, &unloadProc);
}

extern "C" void poseidonModuleUninit(){
	LOG_FATAL("poseidonModuleUninit()");
}
