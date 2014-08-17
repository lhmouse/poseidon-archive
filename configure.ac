AC_PREREQ([2.67])
AC_INIT([poseidon], [0.1], [lh_mouse@126.com])
AC_CONFIG_SRCDIR([src])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_FILES([Makefile])
AC_CONFIG_MACRO_DIR([m4])
AM_INIT_AUTOMAKE

AC_PROG_CC
AC_PROG_CXX
AC_PROG_LIBTOOL

AC_CHECK_LIB([ltdl], [main])
AC_CHECK_LIB([boost_chrono-mt], [main])
AC_CHECK_LIB([boost_thread-mt], [main])
AC_CHECK_LIB([boost_system], [main])
AC_CHECK_LIB([mysqlcppconn], [main])

AC_OUTPUT
