#ifndef INCLUDE_READER_HPP
#define INCLUDE_READER_HPP
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

#include "common.h"
#include "GemmTestCase.hpp"

template<typename T>
class Reader{
public:
    Reader() = default;
    Reader(const char* path){
        ErrCheckExt(nullptr != path, true);
        std::ifstream file(path, std::ios::in);
        read(file);
        file.close();
    }

    ~Reader(){}

    Reader(const Reader&) = delete;
    Reader& operator=(const Reader&) = delete;


    void setPath(const char* path){
        ErrCheckExt(nullptr != path, true);
        std::ifstream file(path, std::ios::in);
        read(file);
        file.close();
    }


    const std::vector<GemmTestCase<T>>& data() const{
        return _data;
    }


private:
    std::vector<GemmTestCase<T>> _data;


    void read(std::ifstream& file){
        ErrCheckExt(file.is_open(), true);

        std::string line_str;
        std::stringstream ss;
        
        GemmTestCase<T> test_case;
        while(std::getline(file, line_str)){
            ss << line_str;
            ss >> test_case.m >> test_case.n >> test_case.k >> \
                    test_case.lda >> test_case.ldb >> test_case.ldc >> \
                    test_case.alpha >> test_case.beta;
            _data.push_back(test_case);

            ss.clear();
        }
    }
};


#endif // INCLUDE_READER_HPP