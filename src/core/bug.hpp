#pragma once

#include "compass.hpp"
#include "landscape.hpp"
#include "path_integration.hpp"
#include "ring_attractor.hpp"

#include <cmath>
#include <memory>
#include <random>
#include <vector>

namespace mayajiva {

/// State snapshot for trajectory recording.
struct BugState {
    double x, y;
    double heading;
    double estimated_heading;
    double bump_amplitude;
};

/// Parameters for Bug construction (at namespace scope for GCC 11 compat).
struct BugParams {
    double x0 = 500.0;
    double y0 = 100.0;
    double heading0 = -1.0; // negative → random
    double goal_heading = 3.0 * M_PI / 4.0;
    double speed = 1.0;
    double kappa = 2.0;
    double sigma_theta = 0.1;
    double sigma_xy = 0.05;
    // Compass
    int n_cry = 1000;
    double contrast = 0.15;
    double mean_yield = 0.5;
    double sigma_sensor = 0.02;
    // Ring attractor
    double ra_tau = 0.05;
    double ra_w_exc = 1.5;
    double ra_w_inh = 4.5;
    double ra_g_mag = 2.0;
    double ra_g_omega = 0.5;
    double ra_noise_sigma = 0.01;
    // CPU4
    double cpu4_leak = 0.0;
    double cpu4_gain = 1.0;
    // Random
    unsigned int seed = 0; // 0 → random
};

/// The navigating bug — full agent composing compass, ring attractor, CPU4.
///
/// Equations of motion (Euler–Maruyama):
///   θ_{t+1} = θ_t + κ sin(θ_goal - θ̂_t) Δt + σ_θ √Δt η_θ
///   x_{t+1} = x_t + v cos(θ_t) Δt + σ_x √Δt η_x
///   y_{t+1} = y_t + v sin(θ_t) Δt + σ_y √Δt η_y
class Bug {
  public:
    using Params = BugParams;

    explicit Bug(const Params& p = Params{}) : params_(p) {
        if (p.seed != 0) {
            rng_ = std::mt19937(p.seed);
        } else {
            rng_ = std::mt19937(std::random_device{}());
        }

        x_ = p.x0;
        y_ = p.y0;
        if (p.heading0 >= 0.0) {
            heading_ = p.heading0;
        } else {
            std::uniform_real_distribution<double> dist(0.0, 2.0 * M_PI);
            heading_ = dist(rng_);
        }
        speed_ = p.speed;
        goal_heading_ = p.goal_heading;
        kappa_ = p.kappa;
        sigma_theta_ = p.sigma_theta;
        sigma_xy_ = p.sigma_xy;

        compass_ = std::make_unique<CompassSensor<8>>(
            p.n_cry, p.contrast, p.mean_yield, p.sigma_sensor, &rng_);

        attractor_ = std::make_unique<RingAttractor<8>>(
            p.ra_tau, p.ra_w_exc, p.ra_w_inh, p.ra_g_mag, p.ra_g_omega,
            0.0, 1.0, p.ra_noise_sigma, &rng_);

        cpu4_ = std::make_unique<CPU4<8>>(p.cpu4_leak, p.cpu4_gain);

        // Init attractor bump near actual heading
        attractor_->reset(heading_);

        // Record initial state
        history_.push_back({x_, y_, heading_,
                           attractor_->heading(), attractor_->bump_amplitude()});
    }

    /// Advance the bug by one timestep. Returns true if still in bounds.
    bool step(double dt, const Landscape& landscape) {
        // 1. Read local magnetic field
        auto [mag_dir, mag_intensity, mag_incl] =
            landscape.magnetic_direction(x_, y_);

        // 2. Heading relative to local field
        double relative_heading = heading_ - mag_dir;

        // 3. Read compass sensor
        auto compass_signal = compass_->read(relative_heading);

        // 4. Compute steering command
        double estimated_heading = attractor_->heading() + mag_dir;
        double heading_error = goal_heading_ - estimated_heading;
        double angular_command = kappa_ * std::sin(heading_error);

        // 5. Update ring attractor
        attractor_->step(dt, compass_signal.data(), angular_command);

        // 6. Update path integrator
        cpu4_->update(attractor_->heading(), speed_, dt);

        // 7. Steer: update heading
        std::normal_distribution<double> normal(0.0, 1.0);
        double noise_theta = sigma_theta_ * std::sqrt(dt) * normal(rng_);
        heading_ += angular_command * dt + noise_theta;
        heading_ = std::fmod(heading_, 2.0 * M_PI);
        if (heading_ < 0) heading_ += 2.0 * M_PI;

        // 8. Move: update position
        double noise_x = sigma_xy_ * std::sqrt(dt) * normal(rng_);
        double noise_y = sigma_xy_ * std::sqrt(dt) * normal(rng_);
        x_ += speed_ * std::cos(heading_) * dt + noise_x;
        y_ += speed_ * std::sin(heading_) * dt + noise_y;

        // 9. Record history
        double est_h = attractor_->heading() + mag_dir;
        history_.push_back({x_, y_, heading_, est_h,
                           attractor_->bump_amplitude()});

        return landscape.in_bounds(x_, y_);
    }

    /// Run for a given duration. Returns false if bug went out of bounds.
    bool run(const Landscape& landscape, double duration, double dt = 0.01) {
        int n_steps = static_cast<int>(duration / dt);
        for (int i = 0; i < n_steps; ++i) {
            if (!step(dt, landscape)) return false;
        }
        return true;
    }

    // Accessors
    double x() const { return x_; }
    double y() const { return y_; }
    double heading() const { return heading_; }
    double goal_heading() const { return goal_heading_; }
    double speed() const { return speed_; }
    const std::vector<BugState>& history() const { return history_; }
    const RingAttractor<8>& attractor() const { return *attractor_; }
    const CPU4<8>& cpu4() const { return *cpu4_; }

    double distance_from_start() const {
        double dx = x_ - history_[0].x;
        double dy = y_ - history_[0].y;
        return std::sqrt(dx * dx + dy * dy);
    }

    double mean_heading_error() const {
        double sum = 0.0;
        for (const auto& s : history_) {
            double err = s.heading - goal_heading_;
            err = std::fmod(err + M_PI, 2.0 * M_PI);
            if (err < 0) err += 2.0 * M_PI;
            err -= M_PI;
            sum += std::abs(err);
        }
        return sum / history_.size();
    }

  private:
    Params params_;
    std::mt19937 rng_;
    double x_, y_, heading_;
    double speed_, goal_heading_, kappa_;
    double sigma_theta_, sigma_xy_;
    std::unique_ptr<CompassSensor<8>> compass_;
    std::unique_ptr<RingAttractor<8>> attractor_;
    std::unique_ptr<CPU4<8>> cpu4_;
    std::vector<BugState> history_;
};

} // namespace mayajiva
