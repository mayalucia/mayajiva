#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <nlohmann/json.hpp>

#include "core/landscape.hpp"

#include <cmath>
#include <fstream>
#include <string>

using json = nlohmann::json;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

static json load_reference() {
    std::ifstream f(REFERENCE_DIR "/landscape_reference.json");
    REQUIRE(f.good());
    json j;
    f >> j;
    return j;
}

TEST_CASE("Uniform field matches Python", "[landscape]") {
    auto ref = load_reference();
    auto& uniform = ref["uniform"]["centre"];

    mayajiva::Landscape land(1000.0, 1000.0, 50.0, 0.0,
                             65.0 * M_PI / 180.0);

    auto [dir, intensity, incl] = land.magnetic_direction(500.0, 500.0);

    REQUIRE_THAT(dir, WithinAbs(uniform["direction"].get<double>(), 1e-12));
    REQUIRE_THAT(intensity, WithinRel(uniform["intensity"].get<double>(), 1e-10));
    REQUIRE_THAT(incl, WithinAbs(uniform["inclination"].get<double>(), 1e-12));
}

TEST_CASE("Dipole anomaly matches Python", "[landscape]") {
    auto ref = load_reference();
    auto& dipole = ref["dipole"];

    mayajiva::Landscape land(1000.0, 1000.0, 50.0, 0.0,
                             65.0 * M_PI / 180.0);
    land.add_anomaly(mayajiva::DipoleAnomaly{500.0, 500.0, 5.0, 50.0});

    for (auto& [key, expected] : dipole["results"].items()) {
        // Parse "px,py"
        auto comma = key.find(',');
        double px = std::stod(key.substr(0, comma));
        double py = std::stod(key.substr(comma + 1));

        auto [dir, intensity, incl] = land.magnetic_direction(px, py);
        auto dev = land.direction_deviation(px, py);

        REQUIRE_THAT(dir, WithinAbs(expected["direction"].get<double>(), 1e-10));
        REQUIRE_THAT(intensity, WithinRel(expected["intensity"].get<double>(), 1e-8));
        REQUIRE_THAT(incl, WithinAbs(expected["inclination"].get<double>(), 1e-10));
        REQUIRE_THAT(dev, WithinAbs(expected["deviation"].get<double>(), 1e-10));
    }
}

TEST_CASE("Fault anomaly matches Python", "[landscape]") {
    auto ref = load_reference();
    auto& fault = ref["fault"];

    mayajiva::Landscape land(1000.0, 1000.0, 50.0, 0.0,
                             65.0 * M_PI / 180.0);
    land.add_anomaly(mayajiva::FaultAnomaly{500.0, 500.0, 0.0, 3.0, 50.0});

    for (auto& [key, expected] : fault["results"].items()) {
        auto comma = key.find(',');
        double px = std::stod(key.substr(0, comma));
        double py = std::stod(key.substr(comma + 1));

        auto [dir, intensity, incl] = land.magnetic_direction(px, py);

        REQUIRE_THAT(dir, WithinAbs(expected["direction"].get<double>(), 1e-10));
        REQUIRE_THAT(intensity, WithinRel(expected["intensity"].get<double>(), 1e-8));
        REQUIRE_THAT(incl, WithinAbs(expected["inclination"].get<double>(), 1e-10));
    }
}

TEST_CASE("Gradient anomaly matches Python", "[landscape]") {
    auto ref = load_reference();
    auto& grad = ref["gradient"];

    mayajiva::Landscape land(1000.0, 1000.0, 50.0, 0.0,
                             65.0 * M_PI / 180.0);
    // Gradient ref defaults to centre of landscape
    land.add_anomaly(mayajiva::GradientAnomaly{0.01, 0.0, 500.0, 500.0});

    for (auto& [key, expected] : grad["results"].items()) {
        auto comma = key.find(',');
        double px = std::stod(key.substr(0, comma));
        double py = std::stod(key.substr(comma + 1));

        auto [dir, intensity, incl] = land.magnetic_direction(px, py);

        REQUIRE_THAT(dir, WithinAbs(expected["direction"].get<double>(), 1e-10));
        REQUIRE_THAT(intensity, WithinRel(expected["intensity"].get<double>(), 1e-8));
        REQUIRE_THAT(incl, WithinAbs(expected["inclination"].get<double>(), 1e-10));
    }
}

TEST_CASE("Landscape bounds check", "[landscape]") {
    mayajiva::Landscape land(1000.0, 1000.0);
    REQUIRE(land.in_bounds(500, 500));
    REQUIRE(land.in_bounds(0, 0));
    REQUIRE(land.in_bounds(1000, 1000));
    REQUIRE_FALSE(land.in_bounds(-1, 500));
    REQUIRE_FALSE(land.in_bounds(500, 1001));
}
