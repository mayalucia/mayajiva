#include "landscape_resource.hpp"

#include <cmath>

namespace godot {

LandscapeResource::LandscapeResource()
    : landscape_(sim_width_, sim_height_, B0_, declination_,
                 inclination_deg_ * M_PI / 180.0) {}

void LandscapeResource::set_B0(double p_B0) {
    B0_ = p_B0;
    rebuild();
}

double LandscapeResource::get_B0() const { return B0_; }

void LandscapeResource::set_declination(double p_dec) {
    declination_ = p_dec;
    rebuild();
}

double LandscapeResource::get_declination() const { return declination_; }

void LandscapeResource::set_inclination_deg(double p_incl) {
    inclination_deg_ = p_incl;
    rebuild();
}

double LandscapeResource::get_inclination_deg() const {
    return inclination_deg_;
}

void LandscapeResource::set_sim_width(double p_w) {
    sim_width_ = p_w;
    rebuild();
}

double LandscapeResource::get_sim_width() const { return sim_width_; }

void LandscapeResource::set_sim_height(double p_h) {
    sim_height_ = p_h;
    rebuild();
}

double LandscapeResource::get_sim_height() const { return sim_height_; }

void LandscapeResource::rebuild() {
    landscape_ = mayajiva::Landscape(sim_width_, sim_height_, B0_,
                                     declination_,
                                     inclination_deg_ * M_PI / 180.0);
}

void LandscapeResource::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_B0", "B0"), &LandscapeResource::set_B0);
    ClassDB::bind_method(D_METHOD("get_B0"), &LandscapeResource::get_B0);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "B0"), "set_B0", "get_B0");

    ClassDB::bind_method(D_METHOD("set_declination", "declination"),
                         &LandscapeResource::set_declination);
    ClassDB::bind_method(D_METHOD("get_declination"),
                         &LandscapeResource::get_declination);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "declination"),
                 "set_declination", "get_declination");

    ClassDB::bind_method(D_METHOD("set_inclination_deg", "inclination_deg"),
                         &LandscapeResource::set_inclination_deg);
    ClassDB::bind_method(D_METHOD("get_inclination_deg"),
                         &LandscapeResource::get_inclination_deg);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "inclination_deg"),
                 "set_inclination_deg", "get_inclination_deg");

    ClassDB::bind_method(D_METHOD("set_sim_width", "width"),
                         &LandscapeResource::set_sim_width);
    ClassDB::bind_method(D_METHOD("get_sim_width"),
                         &LandscapeResource::get_sim_width);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sim_width"),
                 "set_sim_width", "get_sim_width");

    ClassDB::bind_method(D_METHOD("set_sim_height", "height"),
                         &LandscapeResource::set_sim_height);
    ClassDB::bind_method(D_METHOD("get_sim_height"),
                         &LandscapeResource::get_sim_height);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sim_height"),
                 "set_sim_height", "get_sim_height");

    ClassDB::bind_method(D_METHOD("rebuild"), &LandscapeResource::rebuild);
}

} // namespace godot
