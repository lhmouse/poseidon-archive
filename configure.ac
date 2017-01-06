AC_INIT([poseidon], [0.1], [lh_mouse@126.com])
AC_CONFIG_SRCDIR([src])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIR([m4])

AC_PROG_CXX
AC_PROG_LIBTOOL

AC_CHECK_LIB([dl], [main], [], [echo "***** FIX THIS ERROR *****"; exit -2;])
AC_CHECK_LIB([anl], [main], [], [echo "***** FIX THIS ERROR *****"; exit -2;])
PKG_CHECK_MODULES([zlib], [zlib])
AC_CHECK_LIB([z], [main], [], [echo "***** FIX THIS ERROR *****"; exit -2;])
PKG_CHECK_MODULES([openssl], [openssl])
AC_CHECK_LIB([ssl], [main], [], [echo "***** FIX THIS ERROR *****"; exit -2;])
AC_CHECK_LIB([mysqlclient], [main], [], [echo "***** FIX THIS ERROR *****"; exit -2;])
PKG_CHECK_MODULES([bson], [libbson-1.0])
AC_CHECK_LIB([bson-1.0], [main], [], [echo "***** FIX THIS ERROR *****"; exit -2;])
PKG_CHECK_MODULES([mongoc], [libmongoc-1.0])
AC_CHECK_LIB([mongoc-1.0], [main], [], [echo "***** FIX THIS ERROR *****"; exit -2;])

AM_INIT_AUTOMAKE
LT_INIT

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
