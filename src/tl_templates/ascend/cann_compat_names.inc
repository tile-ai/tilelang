#include "tl_templates/ascend/cann_compat_names.h"

// Reinclude with TL_CANN_SCOPE set to push_macro or pop_macro.
#define TL_CANN_PRAGMA_IMPL(x) _Pragma(#x)
#define TL_CANN_PRAGMA(x) TL_CANN_PRAGMA_IMPL(x)
#define TL_CANN_SCOPE_NAME(name) TL_CANN_PRAGMA(TL_CANN_SCOPE(#name))
#define TL_CANN_CORE(name) TL_CANN_SCOPE_NAME(asc_##name)
#define TL_CANN_TYPE(name) TL_CANN_CORE(name)
#ifdef TL_CANN_SIMD_SCOPE
#define TL_CANN_SIMD(name) TL_CANN_CORE(name)
#else
#define TL_CANN_SIMD(name)
#endif

TL_CANN_CORE_NAMES(TL_CANN_CORE)
TL_CANN_SIMD_NAMES(TL_CANN_SIMD)
TL_CANN_TYPE_NAMES(TL_CANN_TYPE)

#undef TL_CANN_CORE
#undef TL_CANN_SIMD
#undef TL_CANN_TYPE
#undef TL_CANN_SCOPE_NAME
#undef TL_CANN_PRAGMA
#undef TL_CANN_PRAGMA_IMPL
