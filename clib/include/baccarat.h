#pragma once

#include "basics.h"

#include <array>
#include <stdio.h>

// anonymous namespace to prevent symbol leakage:
namespace {

/**
 * Iterates over the first two cards of either the player or banker.
 *
 * We call lambda only 55 times, instead of 100, by taking advantage
 * of symmetry (it doesn't matter in which order the cards are drawn).
 */
template<typename T, typename Fn>
_HD_ uint64_t triangulate(T &remaining, Fn lambda) {

    // We keep an accumulator to count the number of combinations:
    uint64_t acc = 0;

    // Sample (without replacement) the first card:
    for (int c1 = 0; c1 < 10; c1++) {
        uint64_t x = remaining[c1]; remaining[c1] -= 1;

        // Sample (without replacement) the second card. We assume wlog
        // that c2 >= c1 here, and correct for the double-counting if
        // necessary.
        for (int c2 = c1; c2 < 10; c2++) {
            uint64_t y = remaining[c2] * x; remaining[c2] -= 1;

            if (c1 != c2) {
                // We need to double-count to account for the two possible
                // orderings (c1 < c2) and (c1 > c2).
                y += y;
            }

            // calculate sum modulo 10, without using modulo:
            int total = c1 + c2;
            if (total >= 10) { total -= 10; }

            // here we do the actual work:
            acc += y * lambda(total);

            // return the second card to the shoe
            remaining[c2] += 1;
        }

        // return the first card to the shoe
        remaining[c1] += 1;
    }

    return acc;
}

template<int N = 10, typename T>
_HD_ int getTotalCards(T &remaining) {

    int shoesize = 0;
    for (int i = 0; i < N; i++) {
        shoesize += remaining[i];
    }

    return shoesize;
}

/**
 * The probability of player natural is equal to the probability of
 * banker natural. We return this probability conditional on the
 * multiset of cards remaining in the shoe.
 */
template<typename T>
_HD_ double naturalProbability(T &remaining) {

    int shoesize = getTotalCards(remaining);

    uint64_t t = triangulate(remaining, [&](int twoCardTotal) __attribute__((always_inline)) {

        return (twoCardTotal >= 8);

    });

    return ((double) t) / ((double) (shoesize * (shoesize - 1)));

}


/**
 * Iterates over the first four cards.
 *
 * @param remaining : counts of cards remaining in the shoe (int[10]-like);
 * @param lambda    : a function which maps the current player- and banker-totals
 *                    to the number of combinations resulting in a win (for the
 *                    bet type in which we're interested, e.g. 'tie' or 'small').
 */
template<typename T, typename Fn>
_HD_ uint64_t countCombinations(T &remaining, Fn lambda) {

    // iterate over possibilities for player's first two cards:
    return triangulate(remaining, [&](int playerTotal) __attribute__((always_inline)) {

        // iterate over possibilities for banker's first two cards:
        return triangulate(remaining, [&](int bankerTotal) __attribute__((always_inline)) {

            // count the combinations conditional on the player and banker totals:
            return lambda(playerTotal, bankerTotal);

        });

    });

}

/**
 * Compute the small probability without performing unnecessary computations.
 */
template<typename T>
_HD_ double fastSmallProbability(T &remaining) {

    int shoesize = getTotalCards(remaining);

    uint64_t t = countCombinations(remaining, [&](int pt, int bt) __attribute__((always_inline)) {

        return ((pt > 7) || (bt > 7) || ((pt > 5) && (bt > 5)));

    });

    uint64_t denominator = shoesize * (shoesize - 1);
            denominator *= (shoesize - 2) * (shoesize - 3);

    return ((double) t) / ((double) denominator);

}

template<typename T>
_HD_ double fastPairProbability(T &remaining) {

    int numerator = 0;
    int sum = 0;

    for (int i = 0; i < 13; i++) {
        int j = remaining[i];
        sum += j;
        numerator += (j * (j - 1));
    }

    int denominator = sum * (sum - 1);

    return ((double) numerator) / ((double) denominator);
}

/**
 * Evaluate the number of winning combinations in Baccarat for a
 * specific bet type (e.g. 'banker win', 'player win', or 'tie').
 *
 * @param remaining : counts of cards remaining in the shoe (int[10]-like);
 * @param lambda    : a function which maps the arguments
 *                    (player total, banker total, number of cards drawn)
 *                    to a boolean specifying whether the bet is successful.
 */
template<typename T, typename Fn>
_HD_ uint64_t evaluateBaccarat(T &remaining, Fn lambda, int shoesize) {

    return countCombinations(remaining, [&](int pt, int bt) __attribute__((always_inline)) {

        int result = 0;

        if ((pt > 7) || (bt > 7) || ((pt > 5) && (bt > 5))) {
            // no-one draws any more cards:
            result += lambda(pt, bt, 4) * (shoesize - 4) * (shoesize - 5);
        } else if (pt < 6) {
            // player draws a third card:

            for (int p3 = 0; p3 < 10; p3++) {
                int x = remaining[p3]; remaining[p3] -= 1;

                // final player total:
                int ptt = (p3 + pt); if (ptt >= 10) { ptt -= 10; }

                bool hit;

                // compressed rules for when banker draws a third card:
                if (bt >= 4) {
                    hit = (p3 <= 7) && (p3 >= (2*bt - 6));
                } else if (bt == 3) {
                    hit = (p3 != 8);
                } else {
                    hit = true;
                }

                if (hit) {
                    // banker also draws a third card:

                    for (int b3 = 0; b3 < 10; b3++) {

                        // final banker total:
                        int btt = (b3 + bt); if (btt >= 10) { btt -= 10; }
                        result += lambda(ptt, btt, 6) * x * remaining[b3];
                    }
                } else {
                    result += lambda(ptt, bt, 5) * x * (shoesize - 5);
                }

                remaining[p3] += 1;
            }

        } else {
            // banker draws a third card:

            for (int b3 = 0; b3 < 10; b3++) {
                // final banker total:
                int btt = (b3 + bt); if (btt >= 10) { btt -= 10; }
                result += lambda(pt, btt, 5) * remaining[b3] * (shoesize - 5);
            }
        }

        return result;

    });

}

/**
 * Evaluate the number of winning combinations in Baccarat for a
 * specific bet type (e.g. 'banker win', 'player win', or 'tie').
 *
 * @param remaining : counts of cards remaining in the shoe (int[10]-like);
 * @param lambda    : a function which maps the arguments
 *                    (player total, banker total, number of cards drawn)
 *                    to a boolean specifying whether the bet is successful.
 */
template<typename T, typename Fn>
_HD_ uint64_t baccaratCombinations(T &remaining, Fn lambda) {

    // We precompute the total number of cards in the shoe, which is
    // used to allow us to 'short-circuit' the calculation when the
    // game is terminated with fewer than 6 cards drawn.
    int shoesize = getTotalCards(remaining);

    return evaluateBaccarat(remaining, lambda, shoesize);
}

/**
 * As above, but computes probabilities instead of combinations:
 */
template<typename T, typename Fn>
_HD_ double baccaratProbability(T &remaining, Fn lambda) {

    // We precompute the total number of cards in the shoe, which is
    // used to allow us to 'short-circuit' the calculation when the
    // game is terminated with fewer than 6 cards drawn.
    int shoesize = getTotalCards(remaining);

    uint64_t combinations = evaluateBaccarat(remaining, lambda, shoesize);

    uint64_t totalCombs = shoesize * (shoesize - 1) * (shoesize - 2);
            totalCombs *= (shoesize - 3) * (shoesize - 4) * (shoesize - 5);

    return ((double) combinations) / ((double) totalCombs);
}

template<int N, typename T>
_HD_ int removeRandomCard(T &remaining, uint32_t random_value) {

    int shoesize = getTotalCards<N, T>(remaining);

    uint64_t z = random_value;
    z *= shoesize;

    uint32_t randcard = z >> 32;

    int idx = -1;

    uint32_t x = 0;
    for (int i = 0; i < N; i++) {
        uint32_t y = x + remaining[i];

        if ((x <= randcard) && (randcard < y)) {
            remaining[i] -= 1;
            idx = i;
        }
        x = y;
    }

    return idx;
}

} // anonymous namespace
