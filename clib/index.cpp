#include <napi.h>
#include <string>
#include <stdint.h>
#include "calOdds.h"


Napi::Number calOdds(const Napi::CallbackInfo &info)
{
    Napi::Env env = info.Env();

    Napi::Array usedCards = info[0].As<Napi::Array>();
    std::int8_t result = calculateOdds(usedCards);

    return Napi::Number::New(env, result);
}

Napi::Object Init(Napi::Env env, Napi::Object exports)
{

    exports.Set(
        Napi::String::New(env, "calOdds"), 
        Napi::Function::New(env, calOdds) 
    );

    return exports;
}


NODE_API_MODULE(greet, Init)
