AC_INIT([poseidon], [0.1], [lh_mouse@126.com])
AC_CONFIG_SRCDIR([src])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIR([m4])

AC_PROG_CXX
AC_PROG_LIBTOOL
AC_LIBTOOL_DLOPEN

AC_LANG([C++])

AC_CHECK_LIB([dl], [dlopen], [], [AC_MSG_ERROR([Could not find dynamic library loader])])
AC_CHECK_LIB([z], [deflate], [], [AC_MSG_ERROR([Could not find zlib])])
AC_CHECK_LIB([ssl], [SSL_read], [], [AC_MSG_ERROR([Could not find openssl])])
AC_CHECK_LIB([mysqlclient], [mysql_real_connect], [], [AC_MSG_ERROR([Could not find libmysqlclient])], [-pthread])
AC_CHECK_LIB([bson-1.0], [bson_new], [], [AC_MSG_ERROR([Could not find libbson-1.0])])
AC_CHECK_LIB([mongoc-1.0], [mongoc_uri_new_for_host_port], [], [AC_MSG_ERROR([Could not find libmongoc-1.0])])

AM_INIT_AUTOMAKE
LT_INIT([disable-static], [dlopen])

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
