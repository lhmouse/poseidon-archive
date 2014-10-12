# 简介

波塞冬（Poseidon）是一个游戏服务端框架。

## 支持功能：
- 传输层 TCP 套接字
- 纯文本 HTTP
- WebSocket
- MySQL 数据库

## 服务器底层采用消息机制，一共有四个线程：
- （目录 src/main/singletons/）

文件名 | 功能
:-- | :--
job_dispatcher.hpp | 主消息线程，负责分发消息，处理逻辑。
epoll_daemon.hpp | epoll 线程，负责接收及发送所有套接字（包括 HTTP）的数据。
database_daemon.hpp | 数据库线程，负责读写数据库。
timer_daemon.hpp | 计时器线程，负责调度计时器，在计时器触发时发送消息。

## 最低编译环境
- automake
- autoconf
- libtool
- pkg-config
- gettext
- make
- g++-4.4
- libboost1.42-dev
- libssl-dev
- libmysqlcppconn-dev

## 运行环境
- 至少 Debian Linux Squeeze
- mysql-server-5.1

## 问题反馈
请联系 ```lh_mouse at 126 dot com```。
