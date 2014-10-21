#include "../precompiled.hpp"
#include "csv_parser.hpp"
#include <fstream>
#include "exception.hpp"
#include "log.hpp"
using namespace Poseidon;

namespace {
/*
void load2DVector(std::vector<std::vector<std::string> > &ret, std::ifstream &ifs){
	std::string token;
	char next;
	ifs.get(next);
	while(ifs){
		const char ch = next;
		ifs.get(next);

		//std::str
	}
}
*/
}

void CsvParser::load(const char *file){
/*	m_rows.clear();

	std::vector<OptionalMap> rows;
	loadFromFile<ExceptionThrower>(rows, file);
	m_rows.swap(rows);*/
}

bool CsvParser::loadNoThrow(const char *file){
/*	m_rows.clear();

	std::vector<OptionalMap> rows;
	if(!loadFromFile<Noop>(rows, file)){
		return false;
	}
	m_rows.swap(rows);*/
	return true;
}
