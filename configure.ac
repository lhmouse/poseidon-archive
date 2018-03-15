AC_INIT([poseidon], [0.1], [lh_mouse@126.com])
AC_CONFIG_SRCDIR([src])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIR([m4])

AC_PROG_CXX
AC_LANG([C++])

AM_INIT_AUTOMAKE([foreign subdir-objects])
LT_INIT([disable-static dlopen])

AC_CHECK_LIB([dl], [dlopen], [], [AC_MSG_ERROR([Could not find dynamic library loader.])])
AC_CHECK_LIB([z], [deflate], [], [AC_MSG_ERROR([Could not find zlib.])])
AC_CHECK_LIB([ssl], [SSL_read], [], [AC_MSG_ERROR([Could not find openssl.])])

AC_ARG_ENABLE([mysql], AS_HELP_STRING([--disable-mysql], [disable MySQL support]))
AM_CONDITIONAL([ENABLE_MYSQL], [test "${enable_mysql}" != "no"])
AM_COND_IF([ENABLE_MYSQL], [
	AC_CHECK_LIB([mysqlclient], [mysql_real_connect], [],
		[AC_MSG_ERROR([Could not find libmysqlclient. (Pass --disable-mysql to disable MySQL support.)])], [-pthread])
	AC_DEFINE([ENABLE_MYSQL], [1], [Define to 1 to build the MySQL daemon.])
])

AC_ARG_ENABLE([mongodb], AS_HELP_STRING([--disable-mongodb], [disable MongoDB support]))
AM_CONDITIONAL([ENABLE_MONGODB], [test "${enable_mongodb}" != "no"])
AM_COND_IF([ENABLE_MONGODB], [
	AC_CHECK_LIB([bson-1.0], [bson_new], [],
		[AC_MSG_ERROR([Could not find libbson-1.0. (Pass --disable-mongodb to disable MongoDB support.)])])
	AC_CHECK_LIB([mongoc-1.0], [mongoc_uri_new_for_host_port], [],
		[AC_MSG_ERROR([Could not find libmongoc-1.0. (Pass --disable-mongodb to disable MongoDB support.)])])
	AC_DEFINE([ENABLE_MONGODB], [1], [Define to 1 to build the MongoDB daemon.])
])

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
