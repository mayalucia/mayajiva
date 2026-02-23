#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <nlohmann/json.hpp>

#include "core/ring_attractor.hpp"

#include <cmath>
#include <fstream>

using json = nlohmann::json;
using Catch::Matchers::WithinAbs;

static json load_reference() {
    std::ifstream f(REFERENCE_DIR "/ring_attractor_reference.json");
    REQUIRE(f.good());
    json j;
    f >> j;
    return j;
}

TEST_CASE("RingAttractor bump stability without input", "[ring_attractor]") {
    auto ref = load_reference();
    double dt = ref["dt"].get<double>();
    auto& no_input = ref["no_input"];

    // Create attractor with zero noise, same params as Python
    std::mt19937 rng(123);
    mayajiva::RingAttractor<8> ra(0.05, 1.5, 4.5, 2.0, 0.5, 0.0, 1.0, 0.0, &rng);
    ra.reset(0.0);

    auto& ref_headings = no_input["headings"];
    auto& ref_states = no_input["states"];

    // Check initial state
    for (int i = 0; i < 8; ++i) {
        double expected = ref_states[0][i].get<double>();
        REQUIRE_THAT(ra.state()[i], WithinAbs(expected, 1e-10));
    }

    // Run and check at each step
    for (size_t step = 0; step < 100; ++step) {
        ra.step(dt);
        double expected_h = ref_headings[step + 1].get<double>();
        // Heading should remain near 0 (bump stable)
        // Use generous tolerance since heading wraps around 0/2π
        double h = ra.heading();
        double diff = std::abs(h - expected_h);
        diff = std::min(diff, 2.0 * M_PI - diff);
        REQUIRE(diff < 1e-8);
    }

    // Final state comparison
    for (int i = 0; i < 8; ++i) {
        double expected = ref_states[100][i].get<double>();
        REQUIRE_THAT(ra.state()[i], WithinAbs(expected, 1e-8));
    }
}

TEST_CASE("RingAttractor tracks angular velocity", "[ring_attractor]") {
    auto ref = load_reference();
    double dt = ref["dt"].get<double>();
    auto& omega_data = ref["angular_velocity"];
    double omega = omega_data["omega"].get<double>();

    std::mt19937 rng(123);
    mayajiva::RingAttractor<8> ra(0.05, 1.5, 4.5, 2.0, 0.5, 0.0, 1.0, 0.0, &rng);
    ra.reset(0.0);

    auto& ref_headings = omega_data["headings"];

    for (size_t step = 0; step < 100; ++step) {
        ra.step(dt, nullptr, omega);
    }

    // After 100 steps at ω=1 rad/s, dt=0.01, the bump should have shifted
    double final_heading = ra.heading();
    double ref_final = ref_headings[100].get<double>();

    // Allow some tolerance — the nonlinear dynamics may accumulate small diffs
    double diff = std::abs(final_heading - ref_final);
    diff = std::min(diff, 2.0 * M_PI - diff);
    REQUIRE(diff < 0.05); // within ~3 degrees
}

TEST_CASE("RingAttractor population vector decode", "[ring_attractor]") {
    std::mt19937 rng(42);
    mayajiva::RingAttractor<8> ra(0.05, 1.5, 4.5, 2.0, 0.5, 0.0, 1.0, 0.0, &rng);

    // Reset with bump at π
    ra.reset(M_PI);
    // Let it settle
    for (int i = 0; i < 50; ++i) ra.step(0.01);

    double h = ra.heading();
    double diff = std::abs(h - M_PI);
    diff = std::min(diff, 2.0 * M_PI - diff);
    REQUIRE(diff < 0.3); // within ~17 degrees after settling

    // Bump amplitude should be positive
    REQUIRE(ra.bump_amplitude() > 0.1);
}
