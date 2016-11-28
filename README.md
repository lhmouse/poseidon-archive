### 简介
### Introduction

波塞冬（Poseidon）是一个游戏服务端框架。  
Poseidon is a C++ framework for game server development.  

### 支持功能
### Features

* CBPP  
CBPP 是 Compressed Binary Protocol for Poseidon 的缩写，  
简介参见 src/cbpp/message_base.hpp 中的注释。  
CBPP is short for 'Compressed Binary Protocol for Poseidon'.  
Specification of CBPP can be found in file 'src/cbpp/message_base.hpp'.  
* HTTP  
* WebSocket  
* MySQL  
* MongoDB  

### 编译所需工具
### Tools Required to Build

* automake  
* autoconf  
* libtool  
* pkg-config  
* libssl-dev  
* gettext  
* make  
* g++  
建议 g++ 版本至少为 4.7 以支持 C++11 特性。  
针对 C++98 的支持可能在未来的版本中被移除。  
It is highly recommended that you use at least g++ 4.7 for C++11 features.  
Support for C++98 is deprecated and might be removed in future versions.  
* libboost-dev  

### 运行环境要求
### Rumtime Environment Requirements

* Debian Linux Wheezy  
这是主要被支持的 Linux 发行版，使用其他发行版的 Linux 不保证兼容性。  
旧版 Debian（例如 Squeeze）有已知的严重问题（例如 g++ 4.4 的 bug 导致运行时  
段错误），我们不对此类问题提供支持，请自行解决。  
This is the Linux distribution that we support primarily and we don't  
guarantee full compatibility with other Linux distributions.  
Older Debian versions (e.g. Squeeze) are known to have some serious problems  
(e.g. runtime SIGSEGV due to bugs in g++ 4.4) that we are not willing to  
provide ANY suppport for. It is you that should work around them.  
* MySQL 5.5  
* MongoDB 2.4  
在 third 目录下有 MySQL 和 MongoDB 的驱动更新脚本可用于编译最新的客户端驱动。  
此处列出的是 Debian Jessie 官方源中的服务端版本。  
There are scripts that can be used to fetch and build latest MySQL and MongoDB  
client drivers in 'third' directory. The version numbers here are of  
server-side packages in Debian Jessie sources.  

### IRC channel:

<irc://chat.freenode.net/%23mcfproj>

### 问题反馈
### Reporting Bugs

请联系 lh_mouse at 126 dot com（注明 Poseidon 相关）。  
Please email to lh_mouse at 126 dot com. (Please state your email as related to Poseidon.)  
