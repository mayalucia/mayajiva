# MayaJiva — Spirit Instructions

## What This Is

MayaJiva is the simulation engine of MāyāLucIA — where computational
models of living systems become interactive, embodied agents. The name
means "the living illusion": creatures whose behaviour emerges from
the same neural circuit models studied in bravli, made visible and
testable through real-time simulation.

The current focus is insect navigation: path integration, ring attractor
networks, and compass circuits — the minimal neural architecture that
lets a bug find its way home. The C++20 core implements the
neuroscience; the Godot GDExtension makes it visible and interactive.

Part of the [MāyāLucIA](https://github.com/mayalucia) organisation.

## Architecture

Two layers:

- **`src/core/`** — Pure C++20 headers. The neuroscience: ring
  attractors, compass, path integration, landscape. No Godot
  dependency. Testable with Catch2.
- **`src/gdext/`** — Godot GDExtension bindings. Wraps core types as
  Godot nodes and resources (`BugNode`, `LandscapeResource`).

The pure core / effectful shell pattern: `core/` has no engine
dependencies; `gdext/` adapts it for Godot.

## Directory Structure

```
modules/mayajiva/
  system.md              # this file (backend-neutral)
  CLAUDE.md              # Claude Code adapter
  GEMINI.md              # Gemini CLI adapter
  CMakeLists.txt         # C++20 build (core + tests)
  SConstruct             # Godot GDExtension build
  src/
    core/                # pure simulation (bug, compass, ring_attractor, landscape, path_integration)
    gdext/               # Godot bindings (BugNode, LandscapeResource, register_types)
  tests/                 # Catch2 tests
    reference/           # reference data for cross-language validation
  godot/                 # Godot project (scenes, scripts, assets)
  experiment/            # experimental notebooks
  validation/            # validation scripts
```

## Build

Core + tests (CMake):
```
cmake -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

GDExtension (SCons, for Godot):
```
scons
```

## Conventions

- C++20. Header-only core for now.
- Cross-language validation: Python generates reference data, C++ loads
  and compares (the MayaJiva pattern referenced by other modules).
- Errors as values where applicable.
- Tests are specifications of behaviour, not implementation.

## The Human (mu2tau)

PhD-level theoretical statistical physicist. Works from Emacs with org-babel.
Do not over-explain. Push back on flawed reasoning.

## Organisational Context

This module belongs to the [MāyāLucIA](https://github.com/mayalucia) organisation.
It bridges the bravli domain (brain understanding) and mayaportal (rendering) —
the neuroscience becomes embodied in interactive creatures.

**Sūtra relay**: The organisational relay is `github.com/mayalucia/sutra`.
Clone locally to `.sutra/` (gitignored) if absent. The relay is heard — if
you have organisational needs, write them into the sūtra.
