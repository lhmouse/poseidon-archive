AC_INIT([poseidon], [0.1], [lh_mouse@126.com])
AC_CONFIG_SRCDIR([src])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIR([m4])

AC_PROG_CXX
AC_PROG_LIBTOOL

PKG_CHECK_MODULES([glib], [glib-2.0])
PKG_CHECK_MODULES([mongo_client], [libmongo-client])

AC_CHECK_LIB([dl], [main], [], [exit -2;])
AC_CHECK_LIB([ssl], [main], [], [exit -2;])
AC_CHECK_LIB([mysqlclient], [main], [], [exit -2;])
AC_CHECK_LIB([glib-2.0], [main], [], [exit -2;])
AC_CHECK_LIB([mongo-client], [main], [], [exit -2;])

AM_INIT_AUTOMAKE
LT_INIT

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
