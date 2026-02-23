#include "bug_node.hpp"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/classes/physics_direct_space_state3d.hpp>
#include <godot_cpp/classes/physics_ray_query_parameters3d.hpp>
#include <godot_cpp/classes/world3d.hpp>

#include <cmath>

namespace godot {

BugNode::BugNode() = default;

void BugNode::_ready() {
    rebuild_bug();
}

void BugNode::_physics_process(double delta) {
    if (!running_ || !bug_ || landscape_res_.is_null()) {
        return;
    }

    double dt = delta / steps_per_frame_;
    auto &landscape = landscape_res_->get_landscape();

    for (int i = 0; i < steps_per_frame_; ++i) {
        if (!bug_->step(dt, landscape)) {
            running_ = false;
            UtilityFunctions::print("BugNode: out of bounds, stopping.");
            break;
        }
    }

    set_position(sim_to_godot(bug_->x(), bug_->y()));

    // Orient the node to face the movement direction.
    // Bug heading: 0=East, π/2=North in sim. Godot: -Z is forward by convention.
    // We rotate around Y so the node faces the direction of travel.
    double h = bug_->heading();
    // Sim heading → Godot yaw: heading 0 (East) → rotation -π/2, heading π/2 (North) → rotation 0
    double yaw = -(h - M_PI / 2.0);
    set_rotation(Vector3(0.0, yaw, 0.0));
}

// --- Property accessors ---

void BugNode::set_seed(int p_seed) { seed_ = p_seed; }
int BugNode::get_seed() const { return seed_; }

void BugNode::set_goal_heading(double p_heading) { goal_heading_ = p_heading; }
double BugNode::get_goal_heading() const { return goal_heading_; }

void BugNode::set_speed(double p_speed) { speed_ = p_speed; }
double BugNode::get_speed() const { return speed_; }

void BugNode::set_kappa(double p_kappa) { kappa_ = p_kappa; }
double BugNode::get_kappa() const { return kappa_; }

void BugNode::set_contrast(double p_contrast) { contrast_ = p_contrast; }
double BugNode::get_contrast() const { return contrast_; }

void BugNode::set_sigma_theta(double p_sigma) { sigma_theta_ = p_sigma; }
double BugNode::get_sigma_theta() const { return sigma_theta_; }

void BugNode::set_sigma_xy(double p_sigma) { sigma_xy_ = p_sigma; }
double BugNode::get_sigma_xy() const { return sigma_xy_; }

void BugNode::set_steps_per_frame(int p_steps) { steps_per_frame_ = p_steps; }
int BugNode::get_steps_per_frame() const { return steps_per_frame_; }

void BugNode::set_landscape_resource(const Ref<LandscapeResource> &p_res) {
    landscape_res_ = p_res;
}

Ref<LandscapeResource> BugNode::get_landscape_resource() const {
    return landscape_res_;
}

double BugNode::get_sim_x() const { return bug_ ? bug_->x() : 0.0; }
double BugNode::get_sim_y() const { return bug_ ? bug_->y() : 0.0; }
double BugNode::get_heading() const { return bug_ ? bug_->heading() : 0.0; }
bool BugNode::is_running() const { return running_; }

void BugNode::start() {
    if (!bug_) {
        rebuild_bug();
    }
    running_ = true;
}

void BugNode::stop() { running_ = false; }

// --- Internal ---

void BugNode::rebuild_bug() {
    mayajiva::BugParams p;
    p.seed = static_cast<unsigned int>(seed_);
    p.goal_heading = goal_heading_;
    p.speed = speed_;
    p.kappa = kappa_;
    p.contrast = contrast_;
    p.sigma_theta = sigma_theta_;
    p.sigma_xy = sigma_xy_;

    bug_ = std::make_unique<mayajiva::Bug>(p);
    running_ = false;

    // Set initial position
    set_position(sim_to_godot(bug_->x(), bug_->y()));
}

Vector3 BugNode::sim_to_godot(double sx, double sy) const {
    double gx = (sx - 500.0) * SCALE_X;
    double gz = -(sy - 500.0) * SCALE_Y;
    double gy = query_terrain_height(gx, gz);
    return Vector3(gx, gy, gz);
}

double BugNode::query_terrain_height(double gx, double gz) const {
    // Default: midpoint of terrain elevation range (~5250m)
    static constexpr double FALLBACK_Y = 5250.0;

    if (!is_inside_tree()) {
        return FALLBACK_Y;
    }

    auto *space = get_world_3d()->get_direct_space_state();
    if (!space) {
        return FALLBACK_Y;
    }

    // Ray from high above straight down
    Vector3 from(gx, 10000.0, gz);
    Vector3 to(gx, 0.0, gz);

    Ref<PhysicsRayQueryParameters3D> query = PhysicsRayQueryParameters3D::create(from, to);
    Dictionary result = space->intersect_ray(query);

    if (result.size() > 0) {
        Vector3 hit = result["position"];
        return hit.y + 50.0; // hover 50m above surface
    }
    return FALLBACK_Y;
}

// --- Bindings ---

void BugNode::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_seed", "seed"), &BugNode::set_seed);
    ClassDB::bind_method(D_METHOD("get_seed"), &BugNode::get_seed);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "seed"), "set_seed", "get_seed");

    ClassDB::bind_method(D_METHOD("set_goal_heading", "heading"),
                         &BugNode::set_goal_heading);
    ClassDB::bind_method(D_METHOD("get_goal_heading"),
                         &BugNode::get_goal_heading);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "goal_heading"),
                 "set_goal_heading", "get_goal_heading");

    ClassDB::bind_method(D_METHOD("set_speed", "speed"), &BugNode::set_speed);
    ClassDB::bind_method(D_METHOD("get_speed"), &BugNode::get_speed);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "speed"), "set_speed",
                 "get_speed");

    ClassDB::bind_method(D_METHOD("set_kappa", "kappa"), &BugNode::set_kappa);
    ClassDB::bind_method(D_METHOD("get_kappa"), &BugNode::get_kappa);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "kappa"), "set_kappa",
                 "get_kappa");

    ClassDB::bind_method(D_METHOD("set_contrast", "contrast"),
                         &BugNode::set_contrast);
    ClassDB::bind_method(D_METHOD("get_contrast"), &BugNode::get_contrast);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "contrast"), "set_contrast",
                 "get_contrast");

    ClassDB::bind_method(D_METHOD("set_sigma_theta", "sigma"),
                         &BugNode::set_sigma_theta);
    ClassDB::bind_method(D_METHOD("get_sigma_theta"),
                         &BugNode::get_sigma_theta);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sigma_theta"),
                 "set_sigma_theta", "get_sigma_theta");

    ClassDB::bind_method(D_METHOD("set_sigma_xy", "sigma"),
                         &BugNode::set_sigma_xy);
    ClassDB::bind_method(D_METHOD("get_sigma_xy"), &BugNode::get_sigma_xy);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sigma_xy"), "set_sigma_xy",
                 "get_sigma_xy");

    ClassDB::bind_method(D_METHOD("set_steps_per_frame", "steps"),
                         &BugNode::set_steps_per_frame);
    ClassDB::bind_method(D_METHOD("get_steps_per_frame"),
                         &BugNode::get_steps_per_frame);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "steps_per_frame"),
                 "set_steps_per_frame", "get_steps_per_frame");

    ClassDB::bind_method(D_METHOD("set_landscape_resource", "resource"),
                         &BugNode::set_landscape_resource);
    ClassDB::bind_method(D_METHOD("get_landscape_resource"),
                         &BugNode::get_landscape_resource);
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "landscape_resource",
                              PROPERTY_HINT_RESOURCE_TYPE,
                              "LandscapeResource"),
                 "set_landscape_resource", "get_landscape_resource");

    ClassDB::bind_method(D_METHOD("get_sim_x"), &BugNode::get_sim_x);
    ClassDB::bind_method(D_METHOD("get_sim_y"), &BugNode::get_sim_y);
    ClassDB::bind_method(D_METHOD("get_heading"), &BugNode::get_heading);
    ClassDB::bind_method(D_METHOD("is_running"), &BugNode::is_running);
    ClassDB::bind_method(D_METHOD("start"), &BugNode::start);
    ClassDB::bind_method(D_METHOD("stop"), &BugNode::stop);
}

} // namespace godot
