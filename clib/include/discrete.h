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
}
