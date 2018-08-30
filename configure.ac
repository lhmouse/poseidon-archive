AC_INIT([poseidon], [0.1], [lh_mouse@126.com], [poseidon], [https://github.com/lhmouse/poseidon])
AC_CONFIG_SRCDIR([src])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIR([m4])

AC_PROG_CXX
AC_LANG([C++])

AM_INIT_AUTOMAKE([foreign subdir-objects])
AM_SILENT_RULES([yes])
LT_INIT([disable-static dlopen])

AC_CHECK_LIB([rt], [clock_gettime], [], [AC_MSG_ERROR([Could not find clock library.])])
AC_CHECK_LIB([pthread], [pthread_once], [], [AC_MSG_ERROR([Could not find POSIX thread library.])])
AC_CHECK_LIB([dl], [dlopen], [], [AC_MSG_ERROR([Could not find dynamic library loader.])])
AC_CHECK_LIB([z], [deflate], [], [AC_MSG_ERROR([Could not find zlib.])])
AC_CHECK_LIB([crypto], [ERR_load_CRYPTO_strings], [], [AC_MSG_ERROR([Could not find openssl.])])
AC_CHECK_LIB([ssl], [SSL_read], [], [AC_MSG_ERROR([Could not find openssl.])])

AC_ARG_ENABLE([mysql], AS_HELP_STRING([--disable-mysql], [disable MySQL support]))
AM_CONDITIONAL([enable_mysql], [test "${enable_mysql}" != "no"])
AM_COND_IF([enable_mysql], [
	AC_CHECK_LIB([mysqlclient], [mysql_real_connect], [],
		[AC_MSG_ERROR([Could not find libmysqlclient. (Pass --disable-mysql to disable MySQL support.)])])
	AC_DEFINE([POSEIDON_ENABLE_MYSQL], [1], [Define to 1 to build the MySQL daemon.])
])

AC_ARG_ENABLE([mongodb], AS_HELP_STRING([--disable-mongodb], [disable MongoDB support]))
AM_CONDITIONAL([enable_mongodb], [test "${enable_mongodb}" != "no"])
AM_COND_IF([enable_mongodb], [
	AC_CHECK_LIB([bson-1.0], [bson_new], [],
		[AC_MSG_ERROR([Could not find libbson-1.0. (Pass --disable-mongodb to disable MongoDB support.)])])
	AC_CHECK_LIB([mongoc-1.0], [mongoc_uri_new_for_host_port], [],
		[AC_MSG_ERROR([Could not find libmongoc-1.0. (Pass --disable-mongodb to disable MongoDB support.)])])
	AC_DEFINE([POSEIDON_ENABLE_MONGODB], [1], [Define to 1 to build the MongoDB daemon.])
])

AC_ARG_ENABLE([magic], AS_HELP_STRING([--disable-magic], [disable magic number recognition]))
AM_CONDITIONAL([enable_magic], [test "${enable_magic}" != "no"])
AM_COND_IF([enable_magic], [
	AC_CHECK_LIB([magic], [magic_open], [],
		[AC_MSG_ERROR([Could not find libmagic. (Pass --disable-magic to disable magic number recognition.)])])
	AC_DEFINE([POSEIDON_ENABLE_MAGIC], [1], [Define to 1 to build the Magic daemon.])
])

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
