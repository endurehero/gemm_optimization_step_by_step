#include "include/GemmTest.hpp"
#include "include/reader.hpp"
#include "config.h"
#include <string>
#include <vector>

static bool is_perf = true;
static std::vector<GEMM_ALGO_TYPE> g_AlgoTypeList;

static std::string help_info = \
"The num of parameters is less than the valid value!\n\
[Usage] GEMM dat_path process_flag [store_path] [ALGO...]\n\
        dat_path    : input data file path.\n\
        process_flag: -perf | -verify, -perf: get algos perf, -verify: cmp 2 algos\n\
        store_path  : on -perf mode, this arg must be set to specific perf result store path\n\
        ALGO        : on -perf mode, arg list can contain any number algo after store_path.\n\
                      on -verify mode, arg list must contain 2 algo.\n";

int main(int argc, const char** argv){

    if(argc < 4){
        printf(help_info.c_str());
        return 0;
    }

    Reader<float> reader(argv[1]);
    
    if(0 == strcmp("-verify", argv[2])){
        is_perf = false;
    }

    int algo_start = 3;
    Recorder* recorder = nullptr;
    if(is_perf){
        recorder = new Recorder(argv[3]);
        algo_start = 4;
    }

    for(; algo_start < argc; ++algo_start){
        g_AlgoTypeList.push_back(algoName2Type(argv[algo_start]));
    }

    if(0 == g_AlgoTypeList.size() || (!is_perf && g_AlgoTypeList.size() != 2)){
        printf(help_info.c_str());
        return 0;
    }

    if(is_perf){
        GemmTest tester(3, 10, recorder);
        for(auto type : g_AlgoTypeList){
            tester.perf<float>(type, reader.data());
        }
    }
    else{
        GemmTest tester;
        tester.verify<float>(g_AlgoTypeList[0], g_AlgoTypeList[1], reader.data());
    }

    if(nullptr != recorder){
        delete recorder; recorder = nullptr;
    }
    
    return 0;
}