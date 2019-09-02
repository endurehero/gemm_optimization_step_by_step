#ifndef INCLUDE_GEMM_FACTORY_HPP
#define INCLUDE_GEMM_FACTORY_HPP

#include <memory>
#include <string>
#include <unordered_map>
#include "common.h"
#include "GemmBase.hpp"

template<typename T>
class GemmFactory{
public:
    typedef std::shared_ptr<GemmBase<T>> (*GemmCreator)();

    static void addCreator(int t, GemmCreator func){
        getCreateRepo()[t] = func;
    }
    
    static std::shared_ptr<GemmBase<T>> getGemmAlgo(GEMM_ALGO_TYPE t) {
        int type = static_cast<int>(t);
        ErrCheckExt(0 != getCreateRepo().count(type), true);
        return getCreateRepo()[type]();
    }

    
private:
    static std::unordered_map<int, GemmCreator>& getCreateRepo(){
        static std::unordered_map<int, GemmCreator> ins;
        return ins;    
    }

    CLASS_SINGLETON_DECLARE(GemmFactory);
};

template <typename Dtype>
class GemmRegisterer {
public:
    GemmRegisterer(GEMM_ALGO_TYPE type, typename GemmFactory<Dtype>::GemmCreator func) {
        GemmFactory<Dtype>::addCreator(static_cast<int>(type), func);
    }
};

#define REGISTER_CREATOR(type, func)\
    static GemmRegisterer<float> g_f_##type(type, func<float>)

#define REGISTER_GEMM(type, classname)\
    template<typename T>\
    std::shared_ptr<GemmBase<T>> classname##_create(){\
        static std::shared_ptr<GemmBase<T>> ins(new classname<T>);\
        return ins;\
    }\
    REGISTER_CREATOR(type, classname##_create)


#endif // INCLUDE_GEMM_FACTORY_HPP