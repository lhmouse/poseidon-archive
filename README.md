# The Poseidon Server Framework

![GNU nano for the win!](https://raw.githubusercontent.com/lhmouse/poseidon/master/GNU-nano-FTW.png)

# Features

1. Coroutines (fibers)
2. Native **TCP**, **TLS over TCP** and **UDP** over either **IPv4** or **IPv6**
3. I/O multiplexing basing on **epoll**
4. Configurable add-ons
5. Asynchronous **MySQL** access (optional, WIP)
6. Asynchronous **MongoDB** access (optional, WIP)

# How to build

#### Prerequisite

1. **GCC** (>= **6**)
2. **autoconf**
3. **automake**
4. **libtool**
5. **OpenSSL** (>= **1.1**)
6. **cmake** (only for building **MySQL** and **MongoDB** libraries)

#### Build and install MySQL and MongoDB client libraries

```sh
cd third/
./build_libmysqlclient_deb.sh
./build_libmongoc_deb.sh
cd ..
```

#### Build and install Asteria

```sh
git submodule update --init
cd asteria/
git checkout master
git pull
autoreconf -i
./configure --disable-static
make -j$(nproc)
./makedeb.sh
cd ..
```

#### Build Poseidon

```sh
autoreconf -i
./configure --disable-static
make -j$(nproc)
```

#### Start Poseidon in build tree

```sh
./run.sh
```

#### Start Poseidon within **GDB**

```sh
./run.sh gdb --args
```

#### Install Poseidon and create default configuration file

```sh
./makedeb.sh
sudo cp /usr/local/etc/poseidon/main.template.conf  \
        /usr/local/etc/poseidon/main.conf
```

#### Start installed Poseidon

```sh
poseidon /usr/local/etc/poseidon
```

# Notes

1. **C++14** is required by **Asteria**.
2. Only **Linux** is supported.
3. **OpenSSL 1.1** is required.

# License

BSD 3-Clause License
