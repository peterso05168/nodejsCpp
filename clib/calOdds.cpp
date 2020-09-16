#include <napi.h>
#include <iostream>

#include "./include/shoe_tracker.h"
#include "calOdds.h"

std::int8_t calculateOdds(Napi::Array arr) {

    ShoeTracker<52> st(8);
    int8_t odds = 2;
    try {
        for(uint32_t i = 0; i < arr.Length(); i++) {
            Napi::Value v = arr[i];
            if (v.IsNumber()) {
                st.removeCard(v.ToNumber().Int32Value());
            }
        }
        odds = st.getBigSmallDiscreteOdds();
    }catch (std::invalid_argument iaex){
        if (strcmp(iaex.what(), st.CARDID_SHOULD_BE_BETWEEN_ZERO_TO_FIFTY_ONE) == 0) {
            odds = -1;
        }else if (strcmp(iaex.what(), st.CARD_VAL_NOT_ALLOWED) == 0) {
            odds = -2;
        }else {
            odds = -3;
        }
    }catch (...) {
        odds = -3;
    }

    return odds;
}
