#ifndef TILELANG_SUPPORT_BASE_H_
#define TILELANG_SUPPORT_BASE_H_

#if defined(_WIN32)
#if defined(TILELANG_EXPORTS)
#define TILELANG_DLL __declspec(dllexport)
#else
#define TILELANG_DLL __declspec(dllimport)
#endif
#else
#define TILELANG_DLL __attribute__((visibility("default")))
#endif

#endif
