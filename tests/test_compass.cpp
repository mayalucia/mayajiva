#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <nlohmann/json.hpp>

#include "core/compass.hpp"

#include <cmath>
#include <fstream>
#include <string>

using json = nlohmann::json;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

static json load_reference() {
    std::ifstream f(REFERENCE_DIR "/compass_reference.json");
    REQUIRE(f.good());
    json j;
    f >> j;
    return j;
}

TEST_CASE("singlet_yield matches Python", "[compass]") {
    auto ref = load_reference();
    auto& alphas = ref["singlet_yield"]["alphas"];
    auto& yields_c015 = ref["singlet_yield"]["yields_c015"];
    auto& yields_c001 = ref["singlet_yield"]["yields_c001"];

    for (size_t i = 0; i < alphas.size(); ++i) {
        double alpha = alphas[i].get<double>();
        double expected_015 = yields_c015[i].get<double>();
        double expected_001 = yields_c001[i].get<double>();

        double got_015 = mayajiva::singlet_yield(alpha, 0.15, 0.5);
        double got_001 = mayajiva::singlet_yield(alpha, 0.01, 0.5);

        REQUIRE_THAT(got_015, WithinAbs(expected_015, 1e-12));
        REQUIRE_THAT(got_001, WithinAbs(expected_001, 1e-12));
    }
}

TEST_CASE("CompassSensor noiseless readings match Python", "[compass]") {
    auto ref = load_reference();
    auto& noiseless = ref["noiseless_readings"];

    // Create sensor with zero noise — deterministic
    std::mt19937 rng(0);
    mayajiva::CompassSensor<8> sensor(1000, 0.15, 0.5, 0.0, &rng);

    for (auto& [heading_str, expected_arr] : noiseless.items()) {
        double heading = std::stod(heading_str);
        auto channels = sensor.read(heading);

        for (int c = 0; c < 8; ++c) {
            double expected = expected_arr[c].get<double>();
            // Small residual from bin-boundary rounding (1000 molecules / 8 bins).
            // Python argmin vs C++ loop can assign boundary molecules differently.
            REQUIRE_THAT(channels[c], WithinAbs(expected, 1e-6));
        }
    }
}

TEST_CASE("singlet_yield has correct symmetry", "[compass]") {
    // Φ_S(0) should be maximum (aligned with field)
    // Φ_S(π/2) should be minimum
    double y_0 = mayajiva::singlet_yield(0.0, 0.15, 0.5);
    double y_pi2 = mayajiva::singlet_yield(M_PI / 2.0, 0.15, 0.5);
    double y_pi = mayajiva::singlet_yield(M_PI, 0.15, 0.5);

    REQUIRE(y_0 > y_pi2);
    REQUIRE_THAT(y_0, WithinAbs(y_pi, 1e-12)); // cos 2α period = π
    // Contrast check: (max - min) / mean
    double mean = 0.5;
    double delta = 0.15 * 0.5;
    REQUIRE_THAT(y_0, WithinAbs(mean + delta, 1e-12));
    REQUIRE_THAT(y_pi2, WithinAbs(mean, 1e-12));
}
