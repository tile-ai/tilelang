#include "tl_templates/ascend/cann_compat_names.h"

// Reinclude with TL_CANN_SCOPE set to push_macro or pop_macro.
#define TL_CANN_PRAGMA_IMPL(x) _Pragma(#x)
#define TL_CANN_PRAGMA(x) TL_CANN_PRAGMA_IMPL(x)
#define TL_CANN_SCOPE_NAME(name) TL_CANN_PRAGMA(TL_CANN_SCOPE(#name))
#define TL_CANN_SCOPE_API(name) TL_CANN_SCOPE_NAME(asc_##name)

TL_CANN_CORE_NAMES(TL_CANN_SCOPE_API)
#ifdef TL_CANN_SIMD_SCOPE
TL_CANN_SIMD_NAMES(TL_CANN_SCOPE_API)
#endif
TL_CANN_TYPE_NAMES(TL_CANN_SCOPE_API)

#undef TL_CANN_SCOPE_API
#undef TL_CANN_SCOPE_NAME
#undef TL_CANN_PRAGMA
#undef TL_CANN_PRAGMA_IMPL
