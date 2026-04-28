# TileLang Native Backend Ownership

This tree is the native backend ownership surface used by CMake. Backend CMake
files own source lists, stub libraries, include paths, and compile definitions
for their backend even when some implementation files still live in transitional
locations such as `src/target` or `src/transform`.

Public FFI registration names are intentionally preserved during this staged
move. Physical file relocation should be mechanical once the CMake ownership and
Python backend metadata are covered by tests.
