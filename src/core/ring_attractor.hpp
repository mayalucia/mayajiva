#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <numeric>
#include <random>

namespace mayajiva {

/// Rate-model ring attractor for heading representation.
///
/// N neurons with local cosine excitatory connectivity, global inhibition
/// (Δ7 pathway), compass input with π-ambiguity resolution via double-angle
/// space, and angular velocity input (P-EN equivalent).
template <int N = 8>
class RingAttractor {
  public:
    using State = std::array<double, N>;

    RingAttractor(double tau = 0.05, double w_exc = 1.5, double w_inh = 4.5,
                  double g_mag = 2.0, double g_omega = 0.5,
                  double threshold = 0.0, double r_max = 1.0,
                  double noise_sigma = 0.01, std::mt19937* rng = nullptr)
        : tau_(tau), w_exc_(w_exc), w_inh_(w_inh), g_mag_(g_mag),
          g_omega_(g_omega), threshold_(threshold), r_max_(r_max),
          noise_sigma_(noise_sigma), rng_(rng), owns_rng_(rng == nullptr) {
        if (owns_rng_) {
            rng_ = new std::mt19937(std::random_device{}());
        }
        init_connectivity();
        r_.fill(0.0);
        init_bump();
    }

    ~RingAttractor() {
        if (owns_rng_) delete rng_;
    }

    RingAttractor(const RingAttractor&) = delete;
    RingAttractor& operator=(const RingAttractor&) = delete;
    RingAttractor(RingAttractor&& o) noexcept
        : tau_(o.tau_), w_exc_(o.w_exc_), w_inh_(o.w_inh_),
          g_mag_(o.g_mag_), g_omega_(o.g_omega_), threshold_(o.threshold_),
          r_max_(o.r_max_), noise_sigma_(o.noise_sigma_),
          rng_(o.rng_), owns_rng_(o.owns_rng_),
          theta_(o.theta_), W_exc_(o.W_exc_), r_(o.r_) {
        o.rng_ = nullptr;
        o.owns_rng_ = false;
    }

    /// Advance ring attractor by one timestep.
    void step(double dt, const double* compass_input = nullptr,
              double angular_velocity = 0.0) {
        // Local excitation: W_exc @ r
        State exc{};
        for (int i = 0; i < N; ++i) {
            double s = 0.0;
            for (int j = 0; j < N; ++j)
                s += W_exc_[i][j] * r_[j];
            exc[i] = s;
        }

        // Global inhibition: w_inh * mean(r)
        double mean_r = 0.0;
        for (int i = 0; i < N; ++i) mean_r += r_[i];
        mean_r /= N;
        double inh = w_inh_ * mean_r;

        // Magnetic compass input — error-correcting shift in double-angle space
        State I_mag{};
        if (compass_input) {
            // Centre the input
            double mean_c = 0.0;
            for (int i = 0; i < N; ++i) mean_c += compass_input[i];
            mean_c /= N;

            // Extract heading from compass in double-angle space
            std::complex<double> z_compass(0.0, 0.0);
            for (int i = 0; i < N; ++i) {
                double c = compass_input[i] - mean_c;
                z_compass += c * std::exp(std::complex<double>(0.0, 2.0 * theta_[i]));
            }

            if (std::abs(z_compass) > 1e-10) {
                double double_heading = std::arg(z_compass);
                double double_bump = 2.0 * heading();
                // Error in double-angle space → actual error in [-π/2, π/2]
                double error = std::arg(
                    std::exp(std::complex<double>(0.0, double_heading - double_bump))) / 2.0;
                // Bump gradient: roll(r, 1) - roll(r, -1)
                for (int i = 0; i < N; ++i) {
                    int ip = (i - 1 + N) % N;  // roll right = previous element
                    int im = (i + 1) % N;      // roll left = next element
                    double bump_grad = r_[ip] - r_[im];
                    I_mag[i] = g_mag_ * error * bump_grad;
                }
            }
        }

        // Angular velocity input (P-EN equivalent)
        State I_omega{};
        if (angular_velocity != 0.0) {
            for (int i = 0; i < N; ++i) {
                int ip = (i - 1 + N) % N;
                int im = (i + 1) % N;
                I_omega[i] = g_omega_ * angular_velocity * (r_[ip] - r_[im]);
            }
        }

        // Neural noise
        State noise{};
        if (noise_sigma_ > 0.0) {
            std::normal_distribution<double> dist(0.0, noise_sigma_);
            for (int i = 0; i < N; ++i)
                noise[i] = dist(*rng_);
        }

        // Total drive → piece-wise linear activation → rate dynamics
        for (int i = 0; i < N; ++i) {
            double drive = exc[i] - inh + I_mag[i] + I_omega[i]
                           - threshold_ + noise[i];
            double activated = std::clamp(drive, 0.0, r_max_);
            double dr = (-r_[i] + activated) / tau_;
            r_[i] = std::clamp(r_[i] + dr * dt, 0.0, r_max_);
        }
    }

    /// Estimated heading from population vector decode, in [0, 2π).
    double heading() const {
        std::complex<double> z(0.0, 0.0);
        for (int i = 0; i < N; ++i)
            z += r_[i] * std::exp(std::complex<double>(0.0, theta_[i]));
        if (std::abs(z) < 1e-10) return 0.0;
        double h = std::arg(z);
        if (h < 0) h += 2.0 * M_PI;
        return h;
    }

    /// Peak-to-trough amplitude of the activity bump.
    double bump_amplitude() const {
        double mx = *std::max_element(r_.begin(), r_.end());
        double mn = *std::min_element(r_.begin(), r_.end());
        return mx - mn;
    }

    /// Reset attractor state. If heading >= 0, init bump centred there.
    void reset(double heading = -1.0) {
        r_.fill(0.0);
        if (heading >= 0.0) {
            for (int i = 0; i < N; ++i) {
                double diff = theta_[i] - heading;
                // Wrap to [-π, π]
                diff = std::fmod(diff + M_PI, 2.0 * M_PI);
                if (diff < 0) diff += 2.0 * M_PI;
                diff -= M_PI;
                r_[i] = std::max(0.0, 0.5 * std::cos(diff));
            }
        } else {
            init_bump();
        }
    }

    const State& state() const { return r_; }
    const State& preferred_directions() const { return theta_; }
    int n() const { return N; }

  private:
    void init_connectivity() {
        for (int i = 0; i < N; ++i)
            theta_[i] = 2.0 * M_PI * i / N;

        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                double d = theta_[i] - theta_[j];
                W_exc_[i][j] = w_exc_ * std::max(0.0, std::cos(d));
            }
    }

    void init_bump() {
        std::uniform_int_distribution<int> dist(0, N - 1);
        int idx = dist(*rng_);
        for (int i = 0; i < N; ++i) {
            int d = std::abs(i - idx);
            d = std::min(d, N - d);
            r_[i] = std::max(0.0, 0.5 - 0.15 * d);
        }
    }

    double tau_, w_exc_, w_inh_, g_mag_, g_omega_;
    double threshold_, r_max_, noise_sigma_;
    std::mt19937* rng_;
    bool owns_rng_;
    State theta_;
    std::array<std::array<double, N>, N> W_exc_;
    State r_;
};

} // namespace mayajiva
