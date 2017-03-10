AC_INIT([poseidon-example-udp-server], [0.1], [lh_mouse@126.com])
AC_CONFIG_SRCDIR([.])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIR([m4])

AC_PROG_CXX
AC_PROG_LIBTOOL

AC_CHECK_LIB([poseidon-main], [main], [], [exit -2;])

AM_INIT_AUTOMAKE
LT_INIT

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
