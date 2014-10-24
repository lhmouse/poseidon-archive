#include "precompiled.hpp"
#include "csv_parser.hpp"
#include <fstream>
#include "exception.hpp"
#include "log.hpp"
using namespace Poseidon;

namespace {

bool loadFromFile(std::vector<OptionalMap> &data, const char *file){
	LOG_DEBUG("Loading CSV file: ", file);

	std::ifstream ifs(file);
	if(!ifs){
		LOG_ERROR("Error opening file: ", file);
		return false;
	}

	std::vector<std::vector<std::string> > rows;
	{
		std::vector<std::string> row;
		std::string token;
		bool inQuote = false;
		char ch;
		do {
			if(!ifs.get(ch)){
				ch = '\n';
			}

			if(ch == '\"'){
				if(!inQuote){
					inQuote = true;
				} else if(ifs.peek() != '\"'){
					inQuote = false;
				} else {
					ifs.ignore();
					token.push_back('\"');
				}
			} else if(!inQuote && ((ch == ',') || (ch == '\n'))){
				std::string trimmed;
				const std::size_t begin = token.find_first_not_of(" \t\r\n");
				if(begin != std::string::npos){
					const std::size_t end = token.find_last_not_of(" \t\r\n") + 1;
					token.substr(begin, end - begin).swap(trimmed);
				}
				token.clear();
				row.push_back(STD_MOVE(trimmed));

				if(ch == '\n'){
					rows.push_back(VAL_INIT);
					rows.back().swap(row);
				}
			} else {
				token.push_back(ch);
			}
		} while(ifs);
	}
	if(rows.empty() || rows.front().empty()){
		LOG_ERROR("The first line of a CSV file may not be empty.");
		return false;
	}

	const std::size_t columnCount = rows.front().size();
	std::vector<SharedNtmbs> keys(columnCount);
	for(std::size_t i = 0; i < columnCount; ++i){
		AUTO_REF(key, rows.front().at(i));
		for(std::size_t j = 0; j < i; ++j){
			if(keys.at(j) == key){
				LOG_ERROR("Duplicate key: ", key);
				return false;
			}
		}
		SharedNtmbs::createOwning(key).swap(keys.at(i));
	}
	for(std::size_t i = 1; i < rows.size(); ++i){
		rows.at(i - 1).swap(rows.at(i));
	}
	rows.pop_back();

	{
		std::size_t line = 1;
		std::size_t i = 0;
		while(i < rows.size()){
			AUTO_REF(row, rows.at(i));
			++line;
			if((row.size() == 1) && row.front().empty()){
				for(std::size_t j = i + 1; j < rows.size(); ++j){
					rows.at(j - 1).swap(rows.at(j));
				}
				rows.pop_back();
				continue;
			}
			if(row.size() != columnCount){
				LOG_ERROR("There are ", row.size(), " column(s) on line ", line,
					" but there are ", columnCount, " in the header");
				return false;
			}
			++i;
		}
	}

	const std::size_t rowCount = rows.size();
	data.resize(rowCount);
	for(std::size_t i = 0; i < rowCount; ++i){
		AUTO_REF(row, rows.at(i));
		AUTO_REF(map, data.at(i));
		for(std::size_t j = 0; j < columnCount; ++j){
			map.create(keys.at(j)).swap(row.at(j));
		}
	}

	LOG_DEBUG("Done loading CSV file: ", file);
	return true;
}

}

void CsvParser::load(const char *file){
	m_data.clear();

	std::vector<OptionalMap> data;
	if(!loadFromFile(data, file)){
		DEBUG_THROW(Exception, "Error loading CSV file");
	}
	m_data.swap(data);
}

bool CsvParser::loadNoThrow(const char *file){
	m_data.clear();

	std::vector<OptionalMap> data;
	if(!loadFromFile(data, file)){
		return false;
	}
	m_data.swap(data);
	return true;
}
