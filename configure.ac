AC_INIT([poseidon], [0.1], [lh_mouse@126.com])
AC_CONFIG_SRCDIR([src])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIR([m4])

AC_PROG_CXX
AC_PROG_LIBTOOL

AC_CHECK_LIB([tcmalloc], [main], [], [exit -2;])
AC_CHECK_LIB([dl], [main], [], [exit -2;])
AC_CHECK_LIB([ssl], [main], [], [exit -2;])
AC_CHECK_LIB([boost_thread], [main], [], [exit -2;])
AC_CHECK_LIB([mysqlclient], [main], [], [exit -2;])

AM_INIT_AUTOMAKE
LT_INIT

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
