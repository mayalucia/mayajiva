#pragma once

#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/core/class_db.hpp>

#include "core/landscape.hpp"

namespace godot {

class LandscapeResource : public Resource {
    GDCLASS(LandscapeResource, Resource)

public:
    LandscapeResource();
    ~LandscapeResource() override = default;

    void set_B0(double p_B0);
    double get_B0() const;

    void set_declination(double p_dec);
    double get_declination() const;

    void set_inclination_deg(double p_incl);
    double get_inclination_deg() const;

    void set_sim_width(double p_w);
    double get_sim_width() const;

    void set_sim_height(double p_h);
    double get_sim_height() const;

    /// Rebuild the internal Landscape from current properties.
    void rebuild();

    /// Access the underlying simulation landscape.
    mayajiva::Landscape& get_landscape() { return landscape_; }
    const mayajiva::Landscape& get_landscape() const { return landscape_; }

protected:
    static void _bind_methods();

private:
    double B0_ = 50.0;
    double declination_ = 0.0;
    double inclination_deg_ = 65.0;
    double sim_width_ = 1000.0;
    double sim_height_ = 1000.0;

    mayajiva::Landscape landscape_;
};

} // namespace godot
