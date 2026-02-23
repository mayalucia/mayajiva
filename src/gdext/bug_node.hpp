#pragma once

#include <godot_cpp/classes/node3d.hpp>
#include <godot_cpp/classes/physics_direct_space_state3d.hpp>
#include <godot_cpp/classes/physics_ray_query_parameters3d.hpp>
#include <godot_cpp/classes/world3d.hpp>
#include <godot_cpp/core/class_db.hpp>

#include "core/bug.hpp"
#include "landscape_resource.hpp"

#include <memory>

namespace godot {

class BugNode : public Node3D {
    GDCLASS(BugNode, Node3D)

public:
    BugNode();
    ~BugNode() override = default;

    void _ready() override;
    void _physics_process(double delta) override;

    // --- Exported properties ---

    void set_seed(int p_seed);
    int get_seed() const;

    void set_goal_heading(double p_heading);
    double get_goal_heading() const;

    void set_speed(double p_speed);
    double get_speed() const;

    void set_kappa(double p_kappa);
    double get_kappa() const;

    void set_contrast(double p_contrast);
    double get_contrast() const;

    void set_sigma_theta(double p_sigma);
    double get_sigma_theta() const;

    void set_sigma_xy(double p_sigma);
    double get_sigma_xy() const;

    void set_steps_per_frame(int p_steps);
    int get_steps_per_frame() const;

    void set_landscape_resource(const Ref<LandscapeResource> &p_res);
    Ref<LandscapeResource> get_landscape_resource() const;

    // --- Sim queries ---

    double get_sim_x() const;
    double get_sim_y() const;
    double get_heading() const;
    bool is_running() const;

    void start();
    void stop();

protected:
    static void _bind_methods();

private:
    void rebuild_bug();

    /// Map simulation (x,y) to Godot world coordinates.
    /// Sim: 1000×1000 body-lengths. Terrain: ~18.87km E-W, ~22.26km N-S.
    /// sim_x → Godot X (East):  (sim_x - 500) * scale_x
    /// sim_y → Godot Z (South): -(sim_y - 500) * scale_y
    /// Y = 0 (flat, Phase 1)
    static constexpr double SCALE_X = 18.87;  // meters per BL (East)
    static constexpr double SCALE_Y = 22.26;  // meters per BL (North→South)

    Vector3 sim_to_godot(double sx, double sy) const;
    double query_terrain_height(double gx, double gz) const;

    // Exported params
    int seed_ = 42;
    double goal_heading_ = 3.0 * 3.14159265358979323846 / 4.0; // 3π/4 ≈ SW
    double speed_ = 1.0;
    double kappa_ = 2.0;
    double contrast_ = 0.15;
    double sigma_theta_ = 0.1;
    double sigma_xy_ = 0.05;
    int steps_per_frame_ = 10;

    Ref<LandscapeResource> landscape_res_;
    std::unique_ptr<mayajiva::Bug> bug_;
    bool running_ = false;
};

} // namespace godot
