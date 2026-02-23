#pragma once

#include <array>
#include <cmath>
#include <random>
#include <vector>

namespace mayajiva {

/// Analytical singlet yield: Φ_S(α) = Φ̄_S + (δΦ_S/2)(1 + cos 2α)
inline double singlet_yield(double alpha, double contrast = 0.15,
                            double mean_yield = 0.5) {
    double delta = contrast * mean_yield;
    return mean_yield + 0.5 * delta * (1.0 + std::cos(2.0 * alpha));
}

/// Array of oriented cryptochrome molecules forming a compass sensor.
///
/// N_cry molecules uniformly distributed in orientation are binned into
/// n_channels.  Each read() returns population-averaged, noisy singlet
/// yield per channel.
template <int N_CHANNELS = 8>
class CompassSensor {
  public:
    using Channels = std::array<double, N_CHANNELS>;

    CompassSensor(int n_cry = 1000, double contrast = 0.15,
                  double mean_yield = 0.5, double sigma_sensor = 0.02,
                  std::mt19937* rng = nullptr)
        : n_cry_(n_cry), contrast_(contrast), mean_yield_(mean_yield),
          sigma_sensor_(sigma_sensor), rng_(rng), owns_rng_(rng == nullptr) {
        if (owns_rng_) {
            rng_ = new std::mt19937(std::random_device{}());
        }
        init_molecules();
    }

    ~CompassSensor() {
        if (owns_rng_) delete rng_;
    }

    // Non-copyable due to RNG ownership
    CompassSensor(const CompassSensor&) = delete;
    CompassSensor& operator=(const CompassSensor&) = delete;
    CompassSensor(CompassSensor&& o) noexcept
        : n_cry_(o.n_cry_), contrast_(o.contrast_),
          mean_yield_(o.mean_yield_), sigma_sensor_(o.sigma_sensor_),
          rng_(o.rng_), owns_rng_(o.owns_rng_),
          phi_(std::move(o.phi_)),
          assignments_(std::move(o.assignments_)),
          channel_counts_(o.channel_counts_) {
        o.rng_ = nullptr;
        o.owns_rng_ = false;
    }

    /// Read compass at given heading (rad, relative to local magnetic field).
    Channels read(double heading) {
        Channels channels{};
        std::array<double, N_CHANNELS> sums{};
        std::normal_distribution<double> noise_dist(0.0, sigma_sensor_);

        for (int k = 0; k < n_cry_; ++k) {
            double alpha = heading - phi_[k];
            double y = singlet_yield(alpha, contrast_, mean_yield_);
            if (sigma_sensor_ > 0.0) {
                y += noise_dist(*rng_);
            }
            sums[assignments_[k]] += y;
        }

        for (int c = 0; c < N_CHANNELS; ++c) {
            if (channel_counts_[c] > 0)
                channels[c] = sums[c] / channel_counts_[c];
        }
        return channels;
    }

    int n_channels() const { return N_CHANNELS; }
    int n_cry() const { return n_cry_; }
    double contrast() const { return contrast_; }
    double mean_yield() const { return mean_yield_; }

  private:
    void init_molecules() {
        static constexpr double TWO_PI = 2.0 * M_PI;
        phi_.resize(n_cry_);
        assignments_.resize(n_cry_);
        channel_counts_.fill(0);

        // Channel centres
        std::array<double, N_CHANNELS> centres;
        for (int c = 0; c < N_CHANNELS; ++c)
            centres[c] = TWO_PI * c / N_CHANNELS;

        // Molecule orientations: uniform on [0, 2π)
        for (int k = 0; k < n_cry_; ++k)
            phi_[k] = TWO_PI * k / n_cry_;

        // Assign each molecule to nearest channel
        for (int k = 0; k < n_cry_; ++k) {
            double best_dist = 1e30;
            int best_c = 0;
            for (int c = 0; c < N_CHANNELS; ++c) {
                double diff = phi_[k] - centres[c];
                // Wrap to [-π, π]
                diff = std::fmod(diff + M_PI, TWO_PI);
                if (diff < 0) diff += TWO_PI;
                diff -= M_PI;
                double d = std::abs(diff);
                if (d < best_dist) {
                    best_dist = d;
                    best_c = c;
                }
            }
            assignments_[k] = best_c;
            channel_counts_[best_c]++;
        }
    }

    int n_cry_;
    double contrast_;
    double mean_yield_;
    double sigma_sensor_;
    std::mt19937* rng_;
    bool owns_rng_;
    std::vector<double> phi_;
    std::vector<int> assignments_;
    std::array<int, N_CHANNELS> channel_counts_;
};

} // namespace mayajiva
