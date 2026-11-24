// Intentionally repeatable, like cann_compat_begin.h.
#if defined(__NPU_ARCH__)
#define TL_CANN_SCOPE pop_macro
#include "tl_templates/ascend/cann_compat_names.inc"
#undef TL_CANN_SCOPE
#endif
