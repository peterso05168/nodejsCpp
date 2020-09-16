#pragma once

#include <string>
#include <vector>

namespace {

std::vector<int> onlyints(std::string &in) {

    std::vector<int> v;

    int val = -1;

    for (auto ch : in) {
        if ((ch >= '0') && (ch <= '9')) {
            if (val < 0) { val = 0; }
            val = val * 10 + (ch - '0');
        } else {
            if (val >= 0) { v.push_back(val); }
            val = -1;
        }
    }

    if (val >= 0) { v.push_back(val); }

    return v;

}

}
