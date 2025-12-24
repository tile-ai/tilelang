if(Z3_FOUND)
    return()
endif()
find_package(Python3 COMPONENTS Interpreter REQUIRED)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import z3; print(z3.__path__[0])"
    OUTPUT_VARIABLE Z3_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE Z3_PYTHON_RESULT
)
if(NOT Z3_PYTHON_RESULT EQUAL 0 OR Z3_PATH STREQUAL "")
    message(FATAL_ERROR "Failed to locate z3 Python package. Ensure z3-solver>=4.13.0 is installed.")
endif()
message("-- Find Z3 in path: ${Z3_PATH}")
find_path(Z3_INCLUDE_DIR NO_DEFAULT_PATH NAMES z3++.h PATHS ${Z3_PATH}/include)
find_library(Z3_LIBRARY NO_DEFAULT_PATH NAMES z3 libz3 PATHS ${Z3_PATH}/bin ${Z3_PATH}/lib ${Z3_PATH}/lib64)
message("-- Found Z3 include dir: ${Z3_INCLUDE_DIR}")
message("-- Found Z3 library: ${Z3_LIBRARY}")
add_library(z3::libz3 SHARED IMPORTED GLOBAL)
set_target_properties(z3::libz3
    PROPERTIES
    IMPORTED_LOCATION ${Z3_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES ${Z3_INCLUDE_DIR}
)
if(NOT Z3_INCLUDE_DIR OR NOT Z3_LIBRARY)
    message(FATAL_ERROR "Could not find Z3 library or include directory")
endif()
set(Z3_CXX_INCLUDE_DIRS ${Z3_INCLUDE_DIR})
set(Z3_C_INCLUDE_DIRS ${Z3_INCLUDE_DIR})
set(Z3_FOUND TRUE)
