#!/usr/bin/env python

env = SConscript("godot-cpp/SConstruct")

# Include paths: src/ for both core/ and gdext/ headers
env.Append(CPPPATH=["src/"])

sources = Glob("src/gdext/*.cpp")

library = env.SharedLibrary(
    "godot/bin/libmayajiva{}{}".format(env["suffix"], env["SHLIBSUFFIX"]),
    source=sources,
)

env.NoCache(library)
Default(library)
