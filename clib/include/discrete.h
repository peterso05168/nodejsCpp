#pragma once

namespace {

enum DiscreteOdds {

    FAR_LEFT,
    CENTRE_LEFT,
    CENTRE,
    CENTRE_RIGHT,
    FAR_RIGHT

};

DiscreteOdds discretise_probability(double smallProb) {

    if (smallProb < 0.3499) {
        return DiscreteOdds::FAR_LEFT;
    } else if (smallProb < 0.3716) {
        return DiscreteOdds::CENTRE_LEFT;
    } else if (smallProb < 0.3852) {
        return DiscreteOdds::CENTRE;
    } else if (smallProb < 0.4039) {
        return DiscreteOdds::CENTRE_RIGHT;
    } else {
        return DiscreteOdds::FAR_RIGHT;
    }

}

// double undiscretise_probability(DiscreteOdds dodds) {

//     double smallProb = 0.0;

//     switch (dodds) {
//         case DiscreteOdds::FAR_LEFT     : smallProb = 0.3356164; break;
//         case DiscreteOdds::CENTRE_LEFT  : smallProb = 0.3642896; break;
//         case DiscreteOdds::CENTRE       : smallProb = 0.3788685; break;
//         case DiscreteOdds::CENTRE_RIGHT : smallProb = 0.3915185; break;
//         case DiscreteOdds::FAR_RIGHT    : smallProb = 0.4162709; break;
//     }

//     return smallProb;

// }


// double probability_to_odds(double p, double edge_pct) {

//     // Convert from percentage to expected value
//     double edge = edge_pct * 0.01;

//     // We have solve linear equation:
//     // -edge = payout_if_win * p + payout_if_loss * (1 - p)
//     // -edge = odds * p - (1 - p)
//     // -edge = (odds + 1) * p - 1
//     // 1 - edge = p * (odds + 1)
//     // odds + 1 = (1 - edge) / p
//     // odds = (1 - edge) / p - 1

//     return (1.0 - edge) / p - 1.0;

// }

// double get_small_odds(DiscreteOdds dodds, double edge_pct = 5.00) {
//     double p = undiscretise_probability(dodds);
//     return probability_to_odds(p, edge_pct);
// }

// double get_big_odds(DiscreteOdds dodds, double edge_pct = 5.00) {
//     double p = undiscretise_probability(dodds);
//     return probability_to_odds(1 - p, edge_pct);
// }

} // anonymous namespace
