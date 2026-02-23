#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <nlohmann/json.hpp>

#include "core/path_integration.hpp"

#include <cmath>
#include <fstream>

using json = nlohmann::json;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

static json load_reference() {
    std::ifstream f(REFERENCE_DIR "/cpu4_reference.json");
    REQUIRE(f.good());
    json j;
    f >> j;
    return j;
}

TEST_CASE("CPU4 north walk matches Python", "[cpu4]") {
    auto ref = load_reference();
    double dt = ref["dt"].get<double>();
    auto& after_north = ref["after_north"];

    mayajiva::CPU4<8> cpu4(0.0, 1.0);

    // Walk North (heading=0) for 100 steps at speed 1
    for (int i = 0; i < 100; ++i)
        cpu4.update(0.0, 1.0, dt);

    // Check memory
    auto& ref_mem = after_north["memory"];
    for (int i = 0; i < 8; ++i) {
        double expected = ref_mem[i].get<double>();
        REQUIRE_THAT(cpu4.memory()[i], WithinAbs(expected, 1e-10));
    }

    // Check home vector
    auto hv = cpu4.home_vector();
    REQUIRE_THAT(hv.distance, WithinRel(after_north["home_distance"].get<double>(), 1e-8));
    REQUIRE_THAT(hv.direction, WithinAbs(after_north["home_direction"].get<double>(), 1e-8));

    // Check displacement
    auto disp = cpu4.displacement();
    REQUIRE_THAT(disp.x, WithinAbs(after_north["displacement"][0].get<double>(), 1e-10));
    REQUIRE_THAT(disp.y, WithinAbs(after_north["displacement"][1].get<double>(), 1e-10));
}

TEST_CASE("CPU4 north then east matches Python", "[cpu4]") {
    auto ref = load_reference();
    double dt = ref["dt"].get<double>();

    mayajiva::CPU4<8> cpu4(0.0, 1.0);

    // North for 100 steps
    for (int i = 0; i < 100; ++i)
        cpu4.update(0.0, 1.0, dt);

    // East (π/2) for 100 steps
    for (int i = 0; i < 100; ++i)
        cpu4.update(M_PI / 2.0, 1.0, dt);

    auto& after_ne = ref["after_north_east"];
    auto hv = cpu4.home_vector();

    REQUIRE_THAT(hv.distance, WithinRel(after_ne["home_distance"].get<double>(), 1e-8));

    // Direction: allow wrapping
    double ref_dir = after_ne["home_direction"].get<double>();
    double diff = std::abs(hv.direction - ref_dir);
    diff = std::min(diff, 2.0 * M_PI - diff);
    REQUIRE(diff < 1e-8);
}

TEST_CASE("CPU4 leaky integrator", "[cpu4]") {
    auto ref = load_reference();
    double dt = ref["dt"].get<double>();
    auto& leaky = ref["leaky"];

    mayajiva::CPU4<8> cpu4(0.1, 1.0);

    for (int i = 0; i < 100; ++i)
        cpu4.update(0.0, 1.0, dt);

    // Leaky memory should be less than perfect integrator
    auto& ref_mem = leaky["memory"];
    for (int i = 0; i < 8; ++i) {
        REQUIRE_THAT(cpu4.memory()[i], WithinAbs(ref_mem[i].get<double>(), 1e-10));
    }

    auto hv = cpu4.home_vector();
    REQUIRE_THAT(hv.distance, WithinRel(leaky["home_distance"].get<double>(), 1e-8));
}

TEST_CASE("CPU4 reset clears memory", "[cpu4]") {
    mayajiva::CPU4<8> cpu4(0.0, 1.0);
    cpu4.update(0.0, 1.0, 0.01);
    cpu4.reset();

    for (int i = 0; i < 8; ++i)
        REQUIRE(cpu4.memory()[i] == 0.0);
}
