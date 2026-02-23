#pragma once

#include <array>
#include <cmath>

namespace mayajiva {

/// CPU4 path integration neurons.
///
/// Velocity integration into distributed population code:
///   m_i(t) += speed × [cos(θ − φ_i)]₊ × dt
/// Optionally leaky: m_i(t+dt) = (1 − λdt) m_i(t) + drive × dt
template <int N = 8>
class CPU4 {
  public:
    CPU4(double leak = 0.0, double gain = 1.0)
        : leak_(leak), gain_(gain) {
        for (int i = 0; i < N; ++i)
            phi_[i] = 2.0 * M_PI * i / N;
        memory_.fill(0.0);
    }

    /// Integrate one timestep of velocity.
    void update(double heading, double speed, double dt) {
        for (int i = 0; i < N; ++i) {
            double cos_val = std::cos(heading - phi_[i]);
            double drive = gain_ * speed * std::max(cos_val, 0.0);
            if (leak_ > 0.0)
                memory_[i] *= (1.0 - leak_ * dt);
            memory_[i] += drive * dt;
        }
    }

    /// Decode home direction and distance.
    /// Returns {distance, direction_to_home}.
    struct HomeVector {
        double distance;
        double direction; // toward home (opposite of displacement)
    };

    HomeVector home_vector() const {
        auto [dx, dy] = displacement();
        double dist = std::sqrt(dx * dx + dy * dy);
        double home_dir = std::atan2(-dy, -dx);
        return {dist, home_dir};
    }

    /// Estimated displacement (x, y) from start.
    struct Vec2 {
        double x, y;
    };

    Vec2 displacement() const {
        double dx = 0.0, dy = 0.0;
        for (int i = 0; i < N; ++i) {
            dx += memory_[i] * std::cos(phi_[i]);
            dy += memory_[i] * std::sin(phi_[i]);
        }
        return {dx, dy};
    }

    void reset() { memory_.fill(0.0); }

    const std::array<double, N>& memory() const { return memory_; }
    int n() const { return N; }

  private:
    double leak_, gain_;
    std::array<double, N> phi_;
    std::array<double, N> memory_;
};

} // namespace mayajiva
