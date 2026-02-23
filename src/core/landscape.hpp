#pragma once

#include <cmath>
#include <string>
#include <variant>
#include <vector>

namespace mayajiva {

// Anomaly types — each is a plain struct

struct GaussianAnomaly {
    double px, py;     // position
    double strength;   // peak field perturbation (μT)
    double radius;     // characteristic radius (BL)
};

struct DipoleAnomaly {
    double px, py;
    double strength;   // peak horizontal anomaly (μT)
    double depth;      // burial depth (BL)
};

struct FaultAnomaly {
    double px, py;     // point on fault
    double azimuth;    // strike from N (rad)
    double contrast;   // field jump (μT)
    double width;      // half-width (BL)
};

struct GradientAnomaly {
    double magnitude;  // μT per BL
    double direction;  // rad from N
    double ref_x, ref_y; // reference point (default: centre)
};

using Anomaly = std::variant<GaussianAnomaly, DipoleAnomaly,
                             FaultAnomaly, GradientAnomaly>;

/// Result of a magnetic field query at a point.
struct FieldResult {
    double direction;   // horizontal field angle from geographic N (rad)
    double intensity;   // horizontal field magnitude (μT)
    double inclination; // local dip angle (rad)
};

/// 2D landscape with configurable magnetic field and anomalies.
class Landscape {
  public:
    Landscape(double width = 1000.0, double height = 1000.0,
              double B0 = 50.0, double declination = 0.0,
              double inclination = 1.1344640137963142) // radians(65)
        : width_(width), height_(height), B0_(B0),
          declination_(declination), inclination_(inclination) {
        B_horizontal_ = B0 * std::cos(inclination);
        B_vertical_ = B0 * std::sin(inclination);
    }

    void add_anomaly(const Anomaly& a) { anomalies_.push_back(a); }
    void clear_anomalies() { anomalies_.clear(); }

    /// Query the local magnetic field at (x, y).
    FieldResult magnetic_direction(double x, double y) const {
        double Bx = B_horizontal_ * std::cos(declination_);
        double By = B_horizontal_ * std::sin(declination_);
        double Bz = B_vertical_;

        for (const auto& anom : anomalies_) {
            auto [dBx, dBy, dBz] = anomaly_perturbation(x, y, anom);
            Bx += dBx;
            By += dBy;
            Bz += dBz;
        }

        double B_h = std::sqrt(Bx * Bx + By * By);
        double dir = std::atan2(By, Bx);
        double local_incl = std::atan2(Bz, B_h);

        return {dir, B_h, local_incl};
    }

    /// Local field direction minus background direction (rad), wrapped to [-π, π].
    double direction_deviation(double x, double y) const {
        auto [dir, intensity, incl] = magnetic_direction(x, y);
        double delta = dir - declination_;
        // Wrap to [-π, π]
        delta = std::fmod(delta + M_PI, 2.0 * M_PI);
        if (delta < 0) delta += 2.0 * M_PI;
        delta -= M_PI;
        return delta;
    }

    bool in_bounds(double x, double y) const {
        return x >= 0 && x <= width_ && y >= 0 && y <= height_;
    }

    double width() const { return width_; }
    double height() const { return height_; }
    double B0() const { return B0_; }
    double declination() const { return declination_; }
    double inclination() const { return inclination_; }

  private:
    struct Vec3 { double x, y, z; };

    Vec3 anomaly_perturbation(double x, double y, const Anomaly& anom) const {
        return std::visit([&](const auto& a) -> Vec3 {
            return perturbation(x, y, a);
        }, anom);
    }

    Vec3 perturbation(double x, double y, const GaussianAnomaly& a) const {
        double dx = x - a.px;
        double dy = y - a.py;
        double r = std::sqrt(dx * dx + dy * dy);
        if (r >= 3.0 * a.radius) return {0, 0, 0};

        double envelope = a.strength * std::exp(-0.5 * (r / a.radius) * (r / a.radius));
        if (r < 1e-6) return {0, 0, 0};

        return {envelope * dx / r, envelope * dy / r, 0.0};
    }

    Vec3 perturbation(double x, double y, const DipoleAnomaly& a) const {
        double dx = x - a.px;
        double dy = y - a.py;
        double rho2 = dx * dx + dy * dy;
        double R2 = rho2 + a.depth * a.depth;
        double R5 = std::pow(R2, 2.5);

        // Normalised so peak horizontal anomaly = strength
        double alpha = a.strength * std::pow(5.0, 2.5) * std::pow(a.depth, 3.0) / 48.0;

        double dBx = alpha * 3.0 * a.depth * dx / R5;
        double dBy = alpha * 3.0 * a.depth * dy / R5;
        double dBz = alpha * (2.0 * a.depth * a.depth - rho2) / R5;

        return {dBx, dBy, dBz};
    }

    Vec3 perturbation(double x, double y, const FaultAnomaly& a) const {
        double d_perp = (x - a.px) * std::sin(a.azimuth)
                      - (y - a.py) * std::cos(a.azimuth);
        double profile = std::tanh(d_perp / a.width);

        double dBx = (a.contrast / 2.0) * profile * std::sin(a.azimuth);
        double dBy = -(a.contrast / 2.0) * profile * std::cos(a.azimuth);
        return {dBx, dBy, 0.0};
    }

    Vec3 perturbation(double x, double y, const GradientAnomaly& a) const {
        double s = (x - a.ref_x) * std::cos(a.direction)
                 + (y - a.ref_y) * std::sin(a.direction);
        double dBx = a.magnitude * s * std::cos(a.direction);
        double dBy = a.magnitude * s * std::sin(a.direction);
        return {dBx, dBy, 0.0};
    }

    double width_, height_, B0_, declination_, inclination_;
    double B_horizontal_, B_vertical_;
    std::vector<Anomaly> anomalies_;
};

} // namespace mayajiva
