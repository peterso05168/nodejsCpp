#pragma once

#include "onlyints.h"
#include "discrete.h"
#include "baccarat.h"
#include <stdexcept>
#include <random>
#include <iostream>

namespace {

    struct ShoeDecoder {

        int games_played;
        int players_unread;
        int cards_unread;

        ShoeDecoder() : games_played(0), players_unread(0), cards_unread(0) { }

    };

    int card2val(int i) {

        int card_value = (i % 13) + 1;

        // Reduce 10s and picture cards to zero:
        if (card_value >= 10) { card_value = 0; }

        return card_value;

    }


    template<int DeckSize = 52, bool verbose = false>
    struct ShoeTracker {

        private:
        ShoeDecoder sd;
        
        public:
        const char* CARDID_SHOULD_BE_BETWEEN_ZERO_TO_FIFTY_ONE = "cardId should be between 0 and 51";
        const char* CARD_VAL_NOT_ALLOWED = "card value occurred more times than is physically possible";

        static std::string card2str(int cardId) {

            std::string res;

            int suit  = cardId / 13;
            int value = cardId % 13;

            if (suit & 1) {
                res = "\033[31;1m"; // red card
            } else {
                res = "\033[1m";  // black card
            }

            switch (value) {
                case 0: res += "A"; break;
                case 1: res += "2"; break;
                case 2: res += "3"; break;
                case 3: res += "4"; break;
                case 4: res += "5"; break;
                case 5: res += "6"; break;
                case 6: res += "7"; break;
                case 7: res += "8"; break;
                case 8: res += "9"; break;
                case 9: res += "10"; break;
                case 10: res += "J"; break;
                case 11: res += "Q"; break;
                case 12: res += "K"; break;
            }

            switch (suit) {
                case 0: res += "♠"; break;
                case 1: res += "♥"; break;
                case 2: res += "♣"; break;
                case 3: res += "♦"; break;
            }

            res += "\033[0m"; // end colour
            return res;

        }

        /**
         * Cards are represented as integers in [0, 51] in the following manner:
         *
         *     | ♠   ♥   ♣   ♦
         * ----+---------------
         *  A  | 0   13  26  39
         *  2  | 1   14  27  40
         *  3  | 2   15  28  41
         *  4  | 3   16  29  42
         *  5  | 4   17  30  43
         *  6  | 5   18  31  44
         *  7  | 6   19  32  45
         *  8  | 7   20  33  46
         *  9  | 8   21  34  47
         *  10 | 9   22  35  48
         *  J  | 10  23  36  49
         *  Q  | 11  24  37  50
         *  K  | 12  25  38  51
         *
         * We store the number of remaining cards in the deck of each type.
         */
        int remaining[DeckSize];
        int number_of_decks;

        int numberOfGamesPlayed() {
            return sd.games_played;
        }

        void reset() {
            for (int i = 0; i < DeckSize; i++) {
                remaining[i] = number_of_decks;
            }
        }

        ShoeTracker(int number_of_decks) : sd(), number_of_decks(number_of_decks) {

            reset();

        }

        void removeCard(int cardId) {

            if ((cardId < 0) || (cardId >= DeckSize)) {
                throw std::invalid_argument(CARDID_SHOULD_BE_BETWEEN_ZERO_TO_FIFTY_ONE);
            }

            if (remaining[cardId] <= 0) {
                throw std::invalid_argument(CARD_VAL_NOT_ALLOWED);
            }

            if (verbose) {
                std::cout << " " << card2str(cardId);
            }

            remaining[cardId] -= 1;
        }

        void decode(int integer) {

            if (sd.cards_unread > 0) {
                removeCard(integer);
                sd.cards_unread -= 1;

                if (verbose && (sd.cards_unread == 0) && (sd.players_unread == 0)) { std::cout << std::endl; }

            } else if (sd.players_unread > 0) {

                if ((integer < 2) || (integer > 3)) {
                    throw std::invalid_argument("number of cards should be either 2 or 3");
                }

                sd.cards_unread = integer;
                sd.players_unread -= 1;
            } else {

                if (integer == 1) {
                    // first game in shoe
                    reset();
                    if (verbose) { std::cout << std::endl; }
                } else if (integer == sd.games_played + 1) {
                    // next game in shoe
                } else {
                    throw std::invalid_argument("game number should be either 1 or the successor of the previous game number");
                }

                sd.games_played = integer;

                if (verbose) { std::cout << "Game " << integer << ":"; }

                sd.players_unread = 2;
            }
        }

        template<typename T>
        void decodeMultiple(T &container) {

            for (auto&& var : container) {
                decode(var);
            }
        }

        void decodeString(std::string &s) {

            auto vec = onlyints(s);
            decodeMultiple(vec);

        }

        std::array<int, 10> getRemainingValueCounts() {

            std::array<int, 10> shoe = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

            for (int i = 0; i < 52; i++) {
                int card_value = card2val(i);
                shoe[card_value] += remaining[i];
            }

            return shoe;
        }

        double getPairExactProbability() {
            std::array<int, 13> shoe;
            for (int i = 0; i < 13; i++) {
                shoe[i] = remaining[i] + remaining[i+13] + remaining[i+26] + remaining[i+39];
            }
            return fastPairProbability(shoe);
        }

        double getSmallExactProbability() {
            auto shoe = getRemainingValueCounts();
            return fastSmallProbability(shoe);
        }

        DiscreteOdds getBigSmallDiscreteOdds() {
            double smallprob = getSmallExactProbability();
            return discretise_probability(smallprob);
        }

    };

}
