#ifndef RECORD_HPP
#define RECORD_HPP
#include <fstream>
#include "common.h"

class Recorder{
public:
    Recorder() = default;

    Recorder(const char* path){
        errCheck(nullptr != path, true);
        file.open(path, std::ios::out);
        errCheck(file.is_open(), true);
    }

    ~Recorder(){
        if(file.is_open()){
            file.close();
        }
    }

    Recorder(const Recorder&) = delete;
    Recorder& operator=(const Recorder&) = delete;


    void setPath(const char* path){
        errCheck(nullptr != path, true);
        errCheck(file.is_open(), false);
        file.open(path, std::ios::out);
        errCheck(file.is_open(), true);
    }

    template<typename T>
    Recorder& operator<<(T s){
        file << s;
        return *this;
    }


private:
    std::fstream file;
};

#endif