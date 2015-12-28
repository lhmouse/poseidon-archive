// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

// 这个文件被置于公有领域（public domain）。

#include "../src/vint50.hpp"
#include <iostream>
#include <iomanip>
#include <string>
#include <cstdlib>

int main(){
	for(;;){
		std::cout <<"Enter an unsigned number: ";
		std::string line;
		if(!std::getline(std::cin, line)){
			break;
		}
		char *eptr;
		const boost::uint64_t val = std::strtoull(line.c_str(), &eptr, 0);
		if(*eptr != 0){
			std::cout <<"  Invalid number: " <<line <<std::endl;
			continue;
		}

		unsigned char bytes[32];
		unsigned char *write = bytes;
		Poseidon::vuint50_to_binary(val, write);
		std::cout <<"  Written: " <<std::hex <<std::uppercase;
		for(const unsigned char *read = bytes; read != write; ++read){
			std::cout <<std::setfill('0') <<std::setw(2)
				<<static_cast<unsigned>(*read) <<' ';
		}
		std::cout <<std::endl;
		std::cout <<"    " <<(write - bytes) <<" byte(s) produced." <<std::endl;
	}
}
