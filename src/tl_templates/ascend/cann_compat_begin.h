// Paired with cann_compat_end.h. Include after SDK headers and before TileLang
// device code. These names must not affect SDK declarations or host code.
#include "tl_templates/ascend/cann_compat.h"

#if defined(__NPU_ARCH__)
#pragma push_macro("asc_copy_gm2ub_align")
#pragma push_macro("asc_copy_ub2gm_align")
#undef asc_copy_gm2ub_align
#undef asc_copy_ub2gm_align
#define asc_copy_gm2ub_align(...) tl::cann::copy_gm2ub_align(__VA_ARGS__)
#define asc_copy_ub2gm_align(...) tl::cann::copy_ub2gm_align(__VA_ARGS__)
#endif
