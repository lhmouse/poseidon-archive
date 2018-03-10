AC_INIT([poseidon], [0.1], [lh_mouse@126.com])
AC_CONFIG_SRCDIR([src])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIR([m4])

AC_PROG_CXX
AC_PROG_LIBTOOL
AC_LIBTOOL_DLOPEN

AC_LANG([C++])

AC_CHECK_LIB([dl], [dlopen], [], [AC_MSG_ERROR([Could not find dynamic library loader])])
PKG_CHECK_MODULES([zlib], [zlib], [], [AC_MSG_ERROR([Could not find zlib])])
PKG_CHECK_MODULES([openssl], [openssl], [], [AC_MSG_ERROR([Could not find openssl])])
AC_CHECK_LIB([mysqlclient], [mysql_real_connect], [], [AC_MSG_ERROR([Could not find libmysqlclient])], [-pthread])
PKG_CHECK_MODULES([bson], [libbson-1.0], [], [AC_MSG_ERROR([Could not find libbson])])
PKG_CHECK_MODULES([mongoc], [libmongoc-1.0], [], [AC_MSG_ERROR([Could not find libmongoc])])

AM_INIT_AUTOMAKE
LT_INIT([disable-static], [dlopen])

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
