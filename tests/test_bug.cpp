#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <nlohmann/json.hpp>

#include "core/bug.hpp"

#include <cmath>
#include <fstream>

using json = nlohmann::json;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

static json load_reference() {
    std::ifstream f(REFERENCE_DIR "/bug_reference.json");
    REQUIRE(f.good());
    json j;
    f >> j;
    return j;
}

TEST_CASE("Bug deterministic trajectory matches Python", "[bug]") {
    auto ref = load_reference();
    double dt = ref["dt"].get<double>();
    int n_steps = ref["n_steps"].get<int>();

    // Set up deterministic bug (zero noise)
    mayajiva::Bug::Params p;
    p.x0 = 500.0;
    p.y0 = 100.0;
    p.heading0 = 0.0;
    p.goal_heading = 3.0 * M_PI / 4.0;
    p.speed = 1.0;
    p.kappa = 2.0;
    p.sigma_theta = 0.0;
    p.sigma_xy = 0.0;
    p.n_cry = 1000;
    p.contrast = 0.15;
    p.sigma_sensor = 0.0;
    p.ra_noise_sigma = 0.0;
    p.seed = 42;

    mayajiva::Landscape land(1000.0, 1000.0, 50.0, 0.0,
                              65.0 * M_PI / 180.0);
    mayajiva::Bug bug(p);

    for (int i = 0; i < n_steps; ++i)
        bug.step(dt, land);

    // Compare trajectory at sampled points
    auto& traj = ref["trajectory"];
    for (auto& [idx_str, expected] : traj.items()) {
        int idx = std::stoi(idx_str);
        const auto& state = bug.history()[idx];

        // Position tolerance: deterministic but ring attractor is nonlinear
        // so we allow modest tolerance for accumulated dynamics differences
        REQUIRE_THAT(state.x, WithinAbs(expected["x"].get<double>(), 0.5));
        REQUIRE_THAT(state.y, WithinAbs(expected["y"].get<double>(), 0.5));

        // Heading: wrap-aware comparison
        double h_diff = std::abs(state.heading - expected["heading"].get<double>());
        h_diff = std::min(h_diff, 2.0 * M_PI - h_diff);
        REQUIRE(h_diff < 0.2); // within ~11 degrees
    }
}

TEST_CASE("Bug steers toward goal heading", "[bug]") {
    // Start heading North (0), goal is SW (3π/4)
    // After enough time, heading should approach goal
    mayajiva::Bug::Params p;
    p.x0 = 500.0;
    p.y0 = 100.0;
    p.heading0 = 0.0;
    p.goal_heading = 3.0 * M_PI / 4.0;
    p.speed = 1.0;
    p.kappa = 2.0;
    p.sigma_theta = 0.0;
    p.sigma_xy = 0.0;
    p.contrast = 0.15;
    p.sigma_sensor = 0.0;
    p.ra_noise_sigma = 0.0;
    p.seed = 99;

    mayajiva::Landscape land(1000.0, 1000.0, 50.0, 0.0,
                              65.0 * M_PI / 180.0);
    mayajiva::Bug bug(p);
    bug.run(land, 10.0, 0.01); // 10 seconds

    // After 10s with kappa=2, the heading should be near goal
    double h = bug.heading();
    double diff = std::abs(h - p.goal_heading);
    diff = std::min(diff, 2.0 * M_PI - diff);
    REQUIRE(diff < 0.5); // within ~30 degrees
}

TEST_CASE("Bug records history", "[bug]") {
    mayajiva::Bug::Params p;
    p.seed = 1;
    p.sigma_theta = 0.0;
    p.sigma_xy = 0.0;
    p.sigma_sensor = 0.0;
    p.ra_noise_sigma = 0.0;

    mayajiva::Landscape land;
    mayajiva::Bug bug(p);

    // Initial state is history[0]
    REQUIRE(bug.history().size() == 1);

    bug.step(0.01, land);
    REQUIRE(bug.history().size() == 2);

    bug.run(land, 0.5, 0.01);
    REQUIRE(bug.history().size() == 52); // 1 initial + 1 step + 50 from run
}
