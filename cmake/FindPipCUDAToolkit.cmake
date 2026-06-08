# FindPipCUDAToolkit.cmake
#
# Locate CUDA toolkit — first trying the host system, then falling back
# to pip-installed packages (nvidia-cuda-nvcc, nvidia-cuda-cccl).
#
# This module should be included BEFORE project() to set CMAKE_CUDA_COMPILER
# when pip CUDA is used.
#
# Detection order:
#   1. Try find_package(CUDAToolkit QUIET) — succeeds if a host CUDA
#      installation is available; skip pip detection.
#   2. If env var WITH_PIP_CUDA_TOOLCHAIN is set to a path (e.g., .../cu13),
#      use that directory directly as the CUDA toolkit root.
#   3. Otherwise, try auto-detecting from the current Python environment's
#      site-packages (works with --no-build-isolation).

function(_tilelang_activate_msvc_env)
  if(NOT WIN32)
    return()
  endif()

  if(NOT CMAKE_GENERATOR MATCHES "Ninja")
    return()
  endif()

  if(DEFINED ENV{VSCMD_VER} AND DEFINED ENV{VCINSTALLDIR})
    return()
  endif()

  set(_vswhere_hints
      "C:/Program Files (x86)/Microsoft Visual Studio/Installer"
      "$ENV{ProgramFiles}/Microsoft Visual Studio/Installer")
  find_program(_TILELANG_VSWHERE
    NAMES vswhere
    PATHS ${_vswhere_hints}
    NO_DEFAULT_PATH)

  if(NOT _TILELANG_VSWHERE)
    return()
  endif()

  execute_process(
    COMMAND "${_TILELANG_VSWHERE}" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    OUTPUT_VARIABLE _tilelang_vs_install
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _tilelang_vswhere_result
  )

  if(NOT _tilelang_vswhere_result EQUAL 0 OR _tilelang_vs_install STREQUAL "")
    return()
  endif()

  set(_tilelang_vsdevcmd "${_tilelang_vs_install}/Common7/Tools/VsDevCmd.bat")
  if(NOT EXISTS "${_tilelang_vsdevcmd}")
    return()
  endif()
  file(TO_NATIVE_PATH "${_tilelang_vsdevcmd}" _tilelang_vsdevcmd_native)

  set(_tilelang_system_path_candidates "")
  foreach(_candidate IN ITEMS
      "$ENV{SystemRoot}/System32"
      "$ENV{SystemRoot}"
      "$ENV{SystemRoot}/System32/Wbem")
    if(NOT "${_candidate}" STREQUAL "" AND EXISTS "${_candidate}")
      list(APPEND _tilelang_system_path_candidates "${_candidate}")
    endif()
  endforeach()
  if(_tilelang_system_path_candidates)
    list(JOIN _tilelang_system_path_candidates ";" _tilelang_system_path_prefix)
    if(DEFINED ENV{PATH} AND NOT "$ENV{PATH}" STREQUAL "")
      set(ENV{PATH} "${_tilelang_system_path_prefix};$ENV{PATH}")
    else()
      set(ENV{PATH} "${_tilelang_system_path_prefix}")
    endif()
  endif()
  set(_tilelang_default_pathext ".COM;.EXE;.BAT;.CMD;.VBS;.VBE;.JS;.JSE;.WSF;.WSH;.MSC")
  if(NOT DEFINED ENV{PATHEXT} OR "$ENV{PATHEXT}" STREQUAL "")
    set(ENV{PATHEXT} "${_tilelang_default_pathext}")
  elseif(NOT "$ENV{PATHEXT}" MATCHES "(^|;)[.]EXE($|;)")
    set(ENV{PATHEXT} "$ENV{PATHEXT};${_tilelang_default_pathext}")
  endif()

  set(_tilelang_target_arch "x64")
  if(CMAKE_GENERATOR_PLATFORM STREQUAL "Win32")
    set(_tilelang_target_arch "x86")
  elseif(CMAKE_GENERATOR_PLATFORM STREQUAL "ARM64")
    set(_tilelang_target_arch "arm64")
  endif()

  set(_tilelang_vsenv_script "${CMAKE_BINARY_DIR}/tilelang-vsenv.bat")
  file(WRITE "${_tilelang_vsenv_script}"
    "@echo off\r\n"
    "call \"${_tilelang_vsdevcmd_native}\" -no_logo -arch=${_tilelang_target_arch} -host_arch=x64 >nul\r\n"
    "if errorlevel 1 exit /b 1\r\n"
    "set PATH\r\n"
    "set INCLUDE\r\n"
    "set LIB\r\n"
    "set LIBPATH\r\n"
    "set VCINSTALLDIR\r\n"
    "set VCToolsInstallDir\r\n"
    "set WindowsSdkDir\r\n"
    "set WindowsSDKVersion\r\n"
    "set UniversalCRTSdkDir\r\n"
    "set UCRTVersion\r\n")
  execute_process(
    COMMAND cmd.exe /d /c call "${_tilelang_vsenv_script}"
    OUTPUT_VARIABLE _tilelang_vs_env
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _tilelang_vsdevcmd_result
  )

  if(NOT _tilelang_vsdevcmd_result EQUAL 0 OR _tilelang_vs_env STREQUAL "")
    return()
  endif()

  string(REPLACE "\r\n" "\n" _tilelang_vs_env "${_tilelang_vs_env}")
  string(REPLACE "\r" "\n" _tilelang_vs_env "${_tilelang_vs_env}")
  string(REPLACE "\n" ";" _tilelang_vs_env_lines "${_tilelang_vs_env}")

  # Snapshot the original PATH so we can merge VsDevCmd's curated MSVC entries
  # in front of the user's PATH. The cmd.exe subprocess that runs VsDevCmd has
  # observably emitted a PATH containing only MSVC dev-env directories on this
  # toolchain, dropping user-level entries (scoop, UniGetUI, etc.). Overwriting
  # ENV{PATH} with that output makes find_program() unable to locate ccache /
  # sccache later in the configure step. Other VsDevCmd-managed variables
  # (INCLUDE / LIB / LIBPATH / VC*) are MSVC-specific and safe to overwrite.
  set(_tilelang_pre_vsdev_path "$ENV{PATH}")

  foreach(_line IN LISTS _tilelang_vs_env_lines)
    if(_line MATCHES "^([^=]+)=(.*)$")
      set(_env_name "${CMAKE_MATCH_1}")
      set(_env_value "${CMAKE_MATCH_2}")
      string(TOUPPER "${_env_name}" _env_name_upper)
      if(_env_name_upper STREQUAL "PATH")
        if(_tilelang_pre_vsdev_path STREQUAL "")
          set(ENV{PATH} "${_env_value}")
        else()
          set(ENV{PATH} "${_env_value};${_tilelang_pre_vsdev_path}")
        endif()
      else()
        set(ENV{${_env_name}} "${_env_value}")
      endif()
    endif()
  endforeach()
  unset(_tilelang_pre_vsdev_path)

  set(CMAKE_C_COMPILER cl CACHE FILEPATH "MSVC C compiler" FORCE)
  set(CMAKE_CXX_COMPILER cl CACHE FILEPATH "MSVC CXX compiler" FORCE)
  set(_tilelang_sdk_arch "${_tilelang_target_arch}")
  if(_tilelang_sdk_arch STREQUAL "arm64")
    set(_tilelang_sdk_arch "arm64")
  elseif(_tilelang_sdk_arch STREQUAL "x86")
    set(_tilelang_sdk_arch "x86")
  else()
    set(_tilelang_sdk_arch "x64")
  endif()
  file(GLOB _tilelang_msvc_tool_dirs LIST_DIRECTORIES true "${_tilelang_vs_install}/VC/Tools/MSVC/*")
  if(NOT _tilelang_msvc_tool_dirs)
    message(FATAL_ERROR "No MSVC tool directories found under ${_tilelang_vs_install}/VC/Tools/MSVC/")
  endif()
  list(SORT _tilelang_msvc_tool_dirs COMPARE NATURAL ORDER DESCENDING)
  list(GET _tilelang_msvc_tool_dirs 0 _tilelang_msvc_tool_dir)
  set(_tilelang_msvc_lib_dir "${_tilelang_msvc_tool_dir}/lib/${_tilelang_sdk_arch}")

  set(_tilelang_winsdk_dir "C:/Program Files (x86)/Windows Kits/10")
  set(_tilelang_winsdk_root "${_tilelang_winsdk_dir}/bin")
  set(_tilelang_winsdk_include_root "${_tilelang_winsdk_dir}/Include")
  set(_tilelang_winsdk_lib_root "${_tilelang_winsdk_dir}/Lib")
  if(DEFINED ENV{WindowsSdkDir} AND NOT "$ENV{WindowsSdkDir}" STREQUAL "")
    file(TO_CMAKE_PATH "$ENV{WindowsSdkDir}" _tilelang_winsdk_dir)
    string(REGEX REPLACE "[/\\\\]+$" "" _tilelang_winsdk_dir "${_tilelang_winsdk_dir}")
    set(_tilelang_winsdk_root "${_tilelang_winsdk_dir}/bin")
    set(_tilelang_winsdk_include_root "${_tilelang_winsdk_dir}/Include")
    set(_tilelang_winsdk_lib_root "${_tilelang_winsdk_dir}/Lib")
  endif()
  set(_tilelang_sdk_bin "")
  set(_tilelang_winsdk_version_dir "")
  set(_tilelang_winsdk_version "")
  file(GLOB _tilelang_winsdk_version_dirs LIST_DIRECTORIES true "${_tilelang_winsdk_root}/*")
  list(SORT _tilelang_winsdk_version_dirs COMPARE NATURAL ORDER DESCENDING)
  foreach(_tilelang_winsdk_version_dir IN LISTS _tilelang_winsdk_version_dirs)
    if(EXISTS "${_tilelang_winsdk_version_dir}/${_tilelang_sdk_arch}/rc.exe")
      set(_tilelang_sdk_bin "${_tilelang_winsdk_version_dir}/${_tilelang_sdk_arch}")
      break()
    endif()
  endforeach()
  message(STATUS "FindPipCUDAToolkit: Windows SDK bin candidate ${_tilelang_sdk_bin}")
  if(EXISTS "${_tilelang_sdk_bin}")
    set(ENV{PATH} "${_tilelang_sdk_bin};$ENV{PATH}")
  endif()
  if(NOT _tilelang_sdk_bin STREQUAL "")
    cmake_path(GET _tilelang_sdk_bin PARENT_PATH _tilelang_winsdk_version_dir)
    cmake_path(GET _tilelang_winsdk_version_dir FILENAME _tilelang_winsdk_version)
    set(_tilelang_winsdk_lib_dir "${_tilelang_winsdk_lib_root}/${_tilelang_winsdk_version}")
  elseif(DEFINED ENV{WindowsSDKVersion} AND NOT "$ENV{WindowsSDKVersion}" STREQUAL "")
    file(TO_CMAKE_PATH "$ENV{WindowsSDKVersion}" _tilelang_winsdk_version)
    string(REGEX REPLACE "[/\\\\]+$" "" _tilelang_winsdk_version "${_tilelang_winsdk_version}")
    set(_tilelang_winsdk_lib_dir "${_tilelang_winsdk_lib_root}/${_tilelang_winsdk_version}")
  else()
    file(GLOB _tilelang_winsdk_include_dirs LIST_DIRECTORIES true "${_tilelang_winsdk_include_root}/*")
    list(SORT _tilelang_winsdk_include_dirs COMPARE NATURAL ORDER DESCENDING)
    if(_tilelang_winsdk_include_dirs)
      list(GET _tilelang_winsdk_include_dirs 0 _tilelang_winsdk_version_dir)
      cmake_path(GET _tilelang_winsdk_version_dir FILENAME _tilelang_winsdk_version)
      set(_tilelang_winsdk_lib_dir "${_tilelang_winsdk_lib_root}/${_tilelang_winsdk_version}")
    endif()
  endif()
  if(NOT DEFINED _tilelang_winsdk_lib_dir)
    set(_tilelang_winsdk_lib_dir "")
  endif()
  if(NOT _tilelang_winsdk_version STREQUAL "")
    set(_tilelang_winsdk_include_dir "${_tilelang_winsdk_include_root}/${_tilelang_winsdk_version}")
  else()
    set(_tilelang_winsdk_include_dir "")
  endif()
  if(NOT _tilelang_winsdk_include_dir STREQUAL "")
    set(_tilelang_include_paths "")
    foreach(_candidate IN ITEMS
        "${_tilelang_msvc_tool_dir}/include"
        "${_tilelang_msvc_tool_dir}/ATLMFC/include"
        "${_tilelang_vs_install}/VC/Auxiliary/VS/include"
        "${_tilelang_winsdk_include_dir}/ucrt"
        "${_tilelang_winsdk_include_dir}/um"
        "${_tilelang_winsdk_include_dir}/shared"
        "${_tilelang_winsdk_include_dir}/winrt"
        "${_tilelang_winsdk_include_dir}/cppwinrt")
      if(EXISTS "${_candidate}")
        list(APPEND _tilelang_include_paths "${_candidate}")
      endif()
    endforeach()
    if(_tilelang_include_paths)
      message(STATUS "FindPipCUDAToolkit: INCLUDE candidates ${_tilelang_include_paths}")
      list(JOIN _tilelang_include_paths ";" _tilelang_include_path)
      if(DEFINED ENV{INCLUDE} AND NOT "$ENV{INCLUDE}" STREQUAL "")
        set(ENV{INCLUDE} "${_tilelang_include_path};$ENV{INCLUDE}")
      else()
        set(ENV{INCLUDE} "${_tilelang_include_path}")
      endif()
      set(_tilelang_include_flags "")
      foreach(_tilelang_include_dir IN LISTS _tilelang_include_paths)
        string(APPEND _tilelang_include_flags " /I\"${_tilelang_include_dir}\"")
      endforeach()
      if(NOT DEFINED _TILELANG_INCLUDE_FLAGS_APPLIED)
        set(CMAKE_C_FLAGS "${_tilelang_include_flags} ${CMAKE_C_FLAGS}" CACHE STRING "C compiler flags" FORCE)
        set(CMAKE_CXX_FLAGS "${_tilelang_include_flags} ${CMAKE_CXX_FLAGS}" CACHE STRING "CXX compiler flags" FORCE)
        set(_TILELANG_INCLUDE_FLAGS_APPLIED TRUE CACHE INTERNAL "")
      endif()
    endif()
    if(NOT DEFINED ENV{WindowsSDKVersion} OR "$ENV{WindowsSDKVersion}" STREQUAL "")
      set(ENV{WindowsSDKVersion} "${_tilelang_winsdk_version}\\")
    endif()
    if(NOT DEFINED ENV{UCRTVersion} OR "$ENV{UCRTVersion}" STREQUAL "")
      set(ENV{UCRTVersion} "${_tilelang_winsdk_version}")
    endif()
    if(NOT DEFINED ENV{WindowsSdkDir} OR "$ENV{WindowsSdkDir}" STREQUAL "")
      set(ENV{WindowsSdkDir} "${_tilelang_winsdk_dir}/")
    endif()
    if(NOT DEFINED ENV{UniversalCRTSdkDir} OR "$ENV{UniversalCRTSdkDir}" STREQUAL "")
      set(ENV{UniversalCRTSdkDir} "${_tilelang_winsdk_dir}/")
    endif()
  endif()
  if(_tilelang_system_path_candidates)
    list(JOIN _tilelang_system_path_candidates ";" _tilelang_system_path_prefix)
    if(DEFINED ENV{PATH} AND NOT "$ENV{PATH}" STREQUAL "")
      set(ENV{PATH} "${_tilelang_system_path_prefix};$ENV{PATH}")
    else()
      set(ENV{PATH} "${_tilelang_system_path_prefix}")
    endif()
  endif()
  if(NOT DEFINED ENV{PATHEXT} OR "$ENV{PATHEXT}" STREQUAL "")
    set(ENV{PATHEXT} "${_tilelang_default_pathext}")
  elseif(NOT "$ENV{PATHEXT}" MATCHES "(^|;)[.]EXE($|;)")
    set(ENV{PATHEXT} "$ENV{PATHEXT};${_tilelang_default_pathext}")
  endif()
  if(NOT _tilelang_sdk_bin STREQUAL "")
    set(ENV{PATH} "${_tilelang_sdk_bin};$ENV{PATH}")
  endif()
  set(_tilelang_lib_paths "")
  foreach(_candidate IN ITEMS
      "${_tilelang_msvc_lib_dir}"
      "${_tilelang_winsdk_lib_dir}/ucrt/${_tilelang_sdk_arch}"
      "${_tilelang_winsdk_lib_dir}/um/${_tilelang_sdk_arch}")
    if(EXISTS "${_candidate}")
      list(APPEND _tilelang_lib_paths "${_candidate}")
    endif()
  endforeach()
  if(_tilelang_lib_paths)
    message(STATUS "FindPipCUDAToolkit: LIBPATH candidates ${_tilelang_lib_paths}")
    list(JOIN _tilelang_lib_paths ";" _tilelang_lib_path)
    if(DEFINED ENV{LIB} AND NOT "$ENV{LIB}" STREQUAL "")
      set(ENV{LIB} "${_tilelang_lib_path};$ENV{LIB}")
    else()
      set(ENV{LIB} "${_tilelang_lib_path}")
    endif()
    set(_tilelang_libpath_flags "")
    foreach(_tilelang_lib_dir IN LISTS _tilelang_lib_paths)
      string(APPEND _tilelang_libpath_flags " /LIBPATH:\"${_tilelang_lib_dir}\"")
    endforeach()
    set(CMAKE_EXE_LINKER_FLAGS_INIT "${_tilelang_libpath_flags} ${CMAKE_EXE_LINKER_FLAGS_INIT}")
    set(CMAKE_SHARED_LINKER_FLAGS_INIT "${_tilelang_libpath_flags} ${CMAKE_SHARED_LINKER_FLAGS_INIT}")
    set(CMAKE_MODULE_LINKER_FLAGS_INIT "${_tilelang_libpath_flags} ${CMAKE_MODULE_LINKER_FLAGS_INIT}")
    set(CMAKE_EXE_LINKER_FLAGS "${_tilelang_libpath_flags} ${CMAKE_EXE_LINKER_FLAGS}" CACHE STRING "Executable linker flags" FORCE)
    set(CMAKE_SHARED_LINKER_FLAGS "${_tilelang_libpath_flags} ${CMAKE_SHARED_LINKER_FLAGS}" CACHE STRING "Shared linker flags" FORCE)
    set(CMAKE_MODULE_LINKER_FLAGS "${_tilelang_libpath_flags} ${CMAKE_MODULE_LINKER_FLAGS}" CACHE STRING "Module linker flags" FORCE)
  endif()
  if(EXISTS "${_tilelang_sdk_bin}/rc.exe")
    set(CMAKE_RC_COMPILER "${_tilelang_sdk_bin}/rc.exe")
    set(CMAKE_RC_COMPILER "${_tilelang_sdk_bin}/rc.exe" CACHE FILEPATH "Windows resource compiler" FORCE)
    set(CMAKE_RC_COMPILER_INIT "${_tilelang_sdk_bin}/rc.exe")
    set(ENV{RC} "${_tilelang_sdk_bin}/rc.exe")
  endif()
  if(EXISTS "${_tilelang_sdk_bin}/mt.exe")
    set(CMAKE_MT "${_tilelang_sdk_bin}/mt.exe")
    set(CMAKE_MT "${_tilelang_sdk_bin}/mt.exe" CACHE FILEPATH "Windows manifest tool" FORCE)
    set(ENV{MT} "${_tilelang_sdk_bin}/mt.exe")
  endif()
  message(STATUS "FindPipCUDAToolkit: RC compiler ${CMAKE_RC_COMPILER}")
  message(STATUS "FindPipCUDAToolkit: MT tool ${CMAKE_MT}")

  # cl.exe is always required as the CUDA host compiler (nvcc on Windows
  # only supports MSVC as host) and as a last-resort fallback for the C/CXX
  # compiler if clang-cl cannot be located.
  find_program(_tilelang_cl_compiler NAMES cl.exe cl)

  # Prefer clang-cl over cl.exe for C/CXX. clang-cl is MSVC-compatible at the
  # command-line level (it accepts /Z7, /FS, /utf-8, /O2, /LD, /link, etc.)
  # and uses the same MSVC headers, libraries, and linker that VsDevCmd just
  # placed on PATH/INCLUDE/LIB above. Empirically clang-cl produces
  # significantly faster code for TileLang's compute-heavy translation units
  # than MSVC at /O2.
  #
  # Set TILELANG_DISABLE_CLANG_CL=1 (env var or -DTILELANG_DISABLE_CLANG_CL=ON)
  # to force the legacy cl.exe path.
  set(_tilelang_disable_clang_cl OFF)
  if(DEFINED ENV{TILELANG_DISABLE_CLANG_CL} AND NOT "$ENV{TILELANG_DISABLE_CLANG_CL}" STREQUAL ""
      AND NOT "$ENV{TILELANG_DISABLE_CLANG_CL}" STREQUAL "0"
      AND NOT "$ENV{TILELANG_DISABLE_CLANG_CL}" STREQUAL "OFF")
    set(_tilelang_disable_clang_cl ON)
  endif()
  if(TILELANG_DISABLE_CLANG_CL)
    set(_tilelang_disable_clang_cl ON)
  endif()

  set(_tilelang_clang_cl "")
  if(NOT _tilelang_disable_clang_cl)
    # Honor explicit user override first.
    if(DEFINED ENV{CLANG_CL} AND EXISTS "$ENV{CLANG_CL}")
      set(_tilelang_clang_cl "$ENV{CLANG_CL}")
    endif()

    if(NOT _tilelang_clang_cl)
      set(_tilelang_clang_cl_hints "")
      # VS-bundled LLVM (when the "C++ Clang Compiler for Windows" workload is
      # installed). VS 2022/2026 places it under VC/Tools/Llvm/{x64,}/bin.
      if(NOT "${_tilelang_vs_install}" STREQUAL "")
        list(APPEND _tilelang_clang_cl_hints
          "${_tilelang_vs_install}/VC/Tools/Llvm/${_tilelang_target_arch}/bin"
          "${_tilelang_vs_install}/VC/Tools/Llvm/x64/bin"
          "${_tilelang_vs_install}/VC/Tools/Llvm/bin")
      endif()
      # Standalone LLVM installation.
      list(APPEND _tilelang_clang_cl_hints
        "C:/Program Files/LLVM/bin"
        "C:/Program Files (x86)/LLVM/bin")
      if(DEFINED ENV{LLVM_HOME} AND NOT "$ENV{LLVM_HOME}" STREQUAL "")
        list(APPEND _tilelang_clang_cl_hints "$ENV{LLVM_HOME}/bin")
      endif()
      if(DEFINED ENV{LLVM_DIR} AND NOT "$ENV{LLVM_DIR}" STREQUAL "")
        list(APPEND _tilelang_clang_cl_hints "$ENV{LLVM_DIR}/bin")
      endif()

      find_program(_tilelang_clang_cl_program
        NAMES clang-cl clang-cl.exe
        HINTS ${_tilelang_clang_cl_hints}
        PATHS ${_tilelang_clang_cl_hints})
      if(_tilelang_clang_cl_program)
        set(_tilelang_clang_cl "${_tilelang_clang_cl_program}")
      endif()
    endif()
  endif()

  if(_tilelang_clang_cl AND EXISTS "${_tilelang_clang_cl}")
    cmake_path(GET _tilelang_clang_cl PARENT_PATH _tilelang_clang_cl_dir)
    # Prepend the LLVM bin dir so lld-link.exe / clang.exe are also found by
    # any toolchain that expects them next to clang-cl.
    if(EXISTS "${_tilelang_clang_cl_dir}")
      set(ENV{PATH} "${_tilelang_clang_cl_dir};$ENV{PATH}")
    endif()
    set(CMAKE_C_COMPILER "${_tilelang_clang_cl}" CACHE FILEPATH "C compiler (clang-cl)" FORCE)
    set(CMAKE_CXX_COMPILER "${_tilelang_clang_cl}" CACHE FILEPATH "CXX compiler (clang-cl)" FORCE)
    # NOTE: do not pre-set CMAKE_C_COMPILER_ID / CMAKE_C_SIMULATE_ID via the
    # cache. Doing so causes CMake to skip the compiler probe and leaves
    # MSVC_VERSION / CMAKE_C_SIMULATE_VERSION unset, which breaks
    # Windows-Clang.cmake when it includes Windows-MSVC.cmake. CMake
    # auto-detects Clang+MSVC-like correctly when the compiler is just a
    # plain path.
    message(STATUS "FindPipCUDAToolkit: preferring clang-cl over MSVC for host C/C++ at ${_tilelang_clang_cl}")
  elseif(_tilelang_cl_compiler)
    set(CMAKE_C_COMPILER "${_tilelang_cl_compiler}" CACHE FILEPATH "MSVC C compiler" FORCE)
    set(CMAKE_CXX_COMPILER "${_tilelang_cl_compiler}" CACHE FILEPATH "MSVC CXX compiler" FORCE)
    if(NOT _tilelang_disable_clang_cl)
      message(STATUS "FindPipCUDAToolkit: clang-cl not found; falling back to MSVC cl.exe for host C/C++")
    endif()
  endif()

  # NVCC on Windows only officially supports MSVC as the host compiler, so
  # always pin CUDA's host compiler to cl.exe regardless of the C/CXX choice.
  if(_tilelang_cl_compiler)
    set(CMAKE_CUDA_HOST_COMPILER "${_tilelang_cl_compiler}" CACHE FILEPATH "CUDA host compiler" FORCE)
    set(ENV{CUDAHOSTCXX} "${_tilelang_cl_compiler}")
  endif()
  message(STATUS "FindPipCUDAToolkit: C compiler ${CMAKE_C_COMPILER}")
  message(STATUS "FindPipCUDAToolkit: CXX compiler ${CMAKE_CXX_COMPILER}")
  message(STATUS "FindPipCUDAToolkit: CUDA host compiler ${CMAKE_CUDA_HOST_COMPILER}")
  find_program(_tilelang_rc_compiler NAMES rc rc.exe)
  if(_tilelang_rc_compiler AND (NOT DEFINED CMAKE_RC_COMPILER OR "${CMAKE_RC_COMPILER}" STREQUAL ""))
    set(CMAKE_RC_COMPILER "${_tilelang_rc_compiler}" CACHE FILEPATH "Windows resource compiler" FORCE)
  endif()
  find_program(_tilelang_mt_program NAMES mt mt.exe)
  if(_tilelang_mt_program AND (NOT DEFINED CMAKE_MT OR "${CMAKE_MT}" STREQUAL ""))
    set(CMAKE_MT "${_tilelang_mt_program}" CACHE FILEPATH "Windows manifest tool" FORCE)
  endif()
  message(STATUS "FindPipCUDAToolkit: activated MSVC environment for Ninja via ${_tilelang_vsdevcmd_native}")
endfunction()

_tilelang_activate_msvc_env()

function(_tilelang_activate_ninja)
  if(NOT CMAKE_GENERATOR MATCHES "Ninja")
    return()
  endif()

  if(DEFINED CMAKE_MAKE_PROGRAM AND NOT "${CMAKE_MAKE_PROGRAM}" STREQUAL "" AND EXISTS "${CMAKE_MAKE_PROGRAM}")
    return()
  endif()

  set(_tilelang_ninja_python_candidates "")
  foreach(_candidate_var IN ITEMS Python3_EXECUTABLE Python_EXECUTABLE PYTHON_EXECUTABLE)
    if(DEFINED ${_candidate_var})
      if(NOT "${${_candidate_var}}" STREQUAL "" AND EXISTS "${${_candidate_var}}")
        list(APPEND _tilelang_ninja_python_candidates "${${_candidate_var}}")
      endif()
    endif()
  endforeach()
  list(REMOVE_DUPLICATES _tilelang_ninja_python_candidates)

  foreach(_python IN LISTS _tilelang_ninja_python_candidates)
    cmake_path(GET _python PARENT_PATH _python_bindir)
    find_program(_tilelang_ninja_program
      NAMES ninja ninja.exe
      PATHS "${_python_bindir}"
      NO_DEFAULT_PATH)
    if(_tilelang_ninja_program)
      set(CMAKE_MAKE_PROGRAM "${_tilelang_ninja_program}" CACHE FILEPATH "Ninja executable" FORCE)
      message(STATUS "FindPipCUDAToolkit: using Ninja from ${_tilelang_ninja_program}")
      return()
    endif()
  endforeach()

  find_program(_tilelang_ninja_program NAMES ninja ninja.exe)
  if(_tilelang_ninja_program)
    set(CMAKE_MAKE_PROGRAM "${_tilelang_ninja_program}" CACHE FILEPATH "Ninja executable" FORCE)
    message(STATUS "FindPipCUDAToolkit: using Ninja from PATH at ${_tilelang_ninja_program}")
  endif()
endfunction()

_tilelang_activate_ninja()

# --- Try host CUDA first ---
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
  set(TILELANG_CUDA_TOOLKIT_AVAILABLE ON CACHE INTERNAL "Whether a CUDA toolkit is available" FORCE)
  set(TILELANG_CUDA_TOOLKIT_SOURCE "host" CACHE INTERNAL "How TileLang discovered CUDA" FORCE)
  return()
endif()

set(TILELANG_CUDA_TOOLKIT_AVAILABLE OFF CACHE INTERNAL "Whether a CUDA toolkit is available" FORCE)
set(TILELANG_CUDA_TOOLKIT_SOURCE "none" CACHE INTERNAL "How TileLang discovered CUDA" FORCE)

set(_PIP_CUDA_PYTHON_CANDIDATES "")

macro(_tilelang_append_python_candidate _candidate)
  if(DEFINED ${_candidate})
    if(NOT "${${_candidate}}" STREQUAL "" AND EXISTS "${${_candidate}}")
      list(APPEND _PIP_CUDA_PYTHON_CANDIDATES "${${_candidate}}")
    endif()
  endif()
endmacro()

foreach(_candidate_var IN ITEMS Python3_EXECUTABLE Python_EXECUTABLE PYTHON_EXECUTABLE)
  if(DEFINED ${_candidate_var})
    _tilelang_append_python_candidate(${_candidate_var})
  endif()
endforeach()

if(DEFINED ENV{VIRTUAL_ENV})
  if(WIN32)
    set(_tilelang_virtualenv_python "$ENV{VIRTUAL_ENV}/Scripts/python.exe")
  else()
    set(_tilelang_virtualenv_python "$ENV{VIRTUAL_ENV}/bin/python")
  endif()
  if(EXISTS "${_tilelang_virtualenv_python}")
    list(APPEND _PIP_CUDA_PYTHON_CANDIDATES "${_tilelang_virtualenv_python}")
  endif()
endif()

if(DEFINED ENV{UV_PROJECT_ENVIRONMENT})
  if(WIN32)
    set(_tilelang_uv_python "$ENV{UV_PROJECT_ENVIRONMENT}/Scripts/python.exe")
  else()
    set(_tilelang_uv_python "$ENV{UV_PROJECT_ENVIRONMENT}/bin/python")
  endif()
  if(EXISTS "${_tilelang_uv_python}")
    list(APPEND _PIP_CUDA_PYTHON_CANDIDATES "${_tilelang_uv_python}")
  endif()
endif()

foreach(_venv_dir IN ITEMS ".venv" "venv")
  if(WIN32)
    set(_tilelang_local_python "${CMAKE_SOURCE_DIR}/${_venv_dir}/Scripts/python.exe")
  else()
    set(_tilelang_local_python "${CMAKE_SOURCE_DIR}/${_venv_dir}/bin/python")
  endif()
  if(EXISTS "${_tilelang_local_python}")
    list(APPEND _PIP_CUDA_PYTHON_CANDIDATES "${_tilelang_local_python}")
  endif()
endforeach()

find_program(_PIP_CUDA_PYTHON_FALLBACK NAMES python3 python)
if(_PIP_CUDA_PYTHON_FALLBACK)
  list(APPEND _PIP_CUDA_PYTHON_CANDIDATES "${_PIP_CUDA_PYTHON_FALLBACK}")
endif()

list(REMOVE_DUPLICATES _PIP_CUDA_PYTHON_CANDIDATES)
if(NOT _PIP_CUDA_PYTHON_CANDIDATES)
  return()
endif()

function(_tilelang_run_find_pip_cuda _out_json _out_python)
  set(_result_json "")
  set(_result_python "")

  foreach(_python IN LISTS _PIP_CUDA_PYTHON_CANDIDATES)
    if(ARGC GREATER 2)
      execute_process(
        COMMAND "${_python}" "${CMAKE_CURRENT_LIST_DIR}/find_pip_cuda.py" "${ARGV2}"
        OUTPUT_VARIABLE _candidate_output
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _candidate_result
      )
    else()
      execute_process(
        COMMAND "${_python}" "${CMAKE_CURRENT_LIST_DIR}/find_pip_cuda.py"
        OUTPUT_VARIABLE _candidate_output
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _candidate_result
      )
    endif()

    if(_candidate_result EQUAL 0)
      set(_result_json "${_candidate_output}")
      set(_result_python "${_python}")
      break()
    endif()
  endforeach()

  set(${_out_json} "${_result_json}" PARENT_SCOPE)
  set(${_out_python} "${_result_python}" PARENT_SCOPE)
endfunction()

# --- Strategy 1: explicit path via env var ---
if(DEFINED ENV{WITH_PIP_CUDA_TOOLCHAIN})
  _tilelang_run_find_pip_cuda(_PIP_CUDA_OUTPUT _PIP_CUDA_PYTHON_EXE "$ENV{WITH_PIP_CUDA_TOOLCHAIN}")
  if(NOT _PIP_CUDA_OUTPUT)
    message(FATAL_ERROR
      "FindPipCUDAToolkit: WITH_PIP_CUDA_TOOLCHAIN is set to '$ENV{WITH_PIP_CUDA_TOOLCHAIN}' "
      "but no pip-installed CUDA toolkit could be resolved from that path")
  endif()
  string(JSON _PIP_CUDA_ROOT GET "${_PIP_CUDA_OUTPUT}" "root")
  message(STATUS "FindPipCUDAToolkit: using env WITH_PIP_CUDA_TOOLCHAIN=${_PIP_CUDA_ROOT}")
else()
  # --- Strategy 2: auto-detect from current Python env ---
  _tilelang_run_find_pip_cuda(_PIP_CUDA_OUTPUT _PIP_CUDA_PYTHON_EXE)
  if(NOT _PIP_CUDA_OUTPUT)
    message(STATUS "FindPipCUDAToolkit: pip-installed CUDA toolkit not found")
    return()
  endif()

  string(JSON _PIP_CUDA_ROOT GET "${_PIP_CUDA_OUTPUT}" "root")
  message(STATUS "FindPipCUDAToolkit: auto-detected from Python environment via ${_PIP_CUDA_PYTHON_EXE}")
endif()

# --- Common pip-CUDA setup ---
string(JSON _PIP_CUDA_NVCC GET "${_PIP_CUDA_OUTPUT}" "nvcc")
string(JSON _PIP_CUDA_LIBRARY_DIR GET "${_PIP_CUDA_OUTPUT}" "library_dir")

set(CMAKE_CUDA_COMPILER "${_PIP_CUDA_NVCC}" CACHE FILEPATH "CUDA compiler (from pip)" FORCE)
set(CUDAToolkit_ROOT "${_PIP_CUDA_ROOT}" CACHE PATH "CUDA toolkit root (from pip)" FORCE)
set(TILELANG_CUDA_TOOLKIT_AVAILABLE ON CACHE INTERNAL "Whether a CUDA toolkit is available" FORCE)
set(TILELANG_CUDA_TOOLKIT_SOURCE "pip" CACHE INTERNAL "How TileLang discovered CUDA" FORCE)

list(APPEND CMAKE_PROGRAM_PATH "${_PIP_CUDA_ROOT}/bin")
if(WIN32)
  list(APPEND CMAKE_PROGRAM_PATH "${_PIP_CUDA_ROOT}/bin/x86_64" "${_PIP_CUDA_ROOT}/nvvm/bin")
  list(APPEND CMAKE_LIBRARY_PATH "${_PIP_CUDA_LIBRARY_DIR}")
  set(ENV{PATH} "${_PIP_CUDA_ROOT}/bin;${_PIP_CUDA_ROOT}/bin/x86_64;${_PIP_CUDA_ROOT}/nvvm/bin;$ENV{PATH}")
else()
  list(APPEND CMAKE_LIBRARY_PATH "${_PIP_CUDA_ROOT}/lib/stubs" "${_PIP_CUDA_LIBRARY_DIR}")
endif()

message(STATUS "FindPipCUDAToolkit: using pip-installed CUDA toolkit")
message(STATUS "  nvcc: ${_PIP_CUDA_NVCC}")
message(STATUS "  root: ${CUDAToolkit_ROOT}")
