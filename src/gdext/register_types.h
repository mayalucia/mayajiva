#pragma once

#include <gdextension_interface.h>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/defs.hpp>
#include <godot_cpp/godot.hpp>

void initialize_mayajiva_module(godot::ModuleInitializationLevel p_level);
void uninitialize_mayajiva_module(godot::ModuleInitializationLevel p_level);
