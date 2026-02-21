import os

shapes = [
    ('32x32b', 'tmem_st_32dp32bNx', 1, 128),
    ('16x64b', 'tmem_st_16dp64bNx', 1, 128),
    ('16x128b', 'tmem_st_16dp128bNx', 2, 64),
    ('16x256b', 'tmem_st_16dp256bNx', 4, 32),
]


def gen_asm_block(shape_str, N, regs_per_x, pack_str):
    total_regs = N * regs_per_x
    reg_refs = ', '.join(f'%{i+1}' for i in range(total_regs))
    instr = f'tcgen05.st.sync.aligned.{shape_str}{pack_str}.x{N}.b32'
    lines = []
    # For large register counts, break the register list
    if total_regs <= 16:
        lines.append(f'      asm volatile("{instr}"')
        lines.append(f'                   "[%0],"')
        lines.append(f'                   "{{{reg_refs}}};\\n"')
    else:
        lines.append(f'      asm volatile(')
        lines.append(f'          "{instr}"')
        lines.append(f'          "[%0],"')
        # Break reg_refs into lines of ~70 chars
        refs = [f'%{i+1}' for i in range(total_regs)]
        ref_lines = []
        cur = ''
        for r in refs:
            if cur:
                test = cur + ', ' + r
            else:
                test = r
            if len(test) > 65:
                ref_lines.append(cur + ',')
                cur = r
            else:
                cur = test
        if cur:
            ref_lines.append(cur)
        for i, rl in enumerate(ref_lines):
            if i == 0 and i == len(ref_lines) - 1:
                lines.append(f'          "{{{rl}}};\\n"')
            elif i == 0:
                lines.append(f'          "{{{rl}"')
            elif i == len(ref_lines) - 1:
                lines.append(f'          "{rl}}};\\n"')
            else:
                lines.append(f'          "{rl}"')

    lines.append(f'                   :')
    # Inputs
    items = ['"r"(dst_addr)'] + [f'"r"(src_ptr[{i}])' for i in range(total_regs)]
    if total_regs <= 4:
        lines.append(f'                   : {", ".join(items)});')
    else:
        input_lines = []
        cur = ''
        for item in items:
            if cur:
                test = cur + ', ' + item
            else:
                test = item
            if len(test) > 60:
                input_lines.append(cur + ',')
                cur = item
            else:
                cur = test
        if cur:
            input_lines.append(cur)
        for i, il in enumerate(input_lines):
            if i == 0:
                pfx = '                   : '
            else:
                pfx = '                     '
            if i == len(input_lines) - 1:
                if il.endswith(','):
                    il = il[:-1]
                lines.append(f'{pfx}{il});')
            else:
                lines.append(f'{pfx}{il}')
    return '\n'.join(lines)


def gen_class(shape_str, class_name, regs_per_x, max_N, is_unpack):
    pack_str = '.unpack::16b' if is_unpack else ''
    tparam = 'true' if is_unpack else 'false'
    lines = []
    lines.append(f'template <> class {class_name}<{tparam}> {{')
    lines.append('public:')
    lines.append('  template <int N>')
    lines.append(f'  static TL_DEVICE void copy(uint32_t const &dst_addr, uint32_t const *src_ptr) {{')
    lines.append(f'    static_assert(N > 0 && (N & (N - 1)) == 0 && N <= {max_N},')
    lines.append(f'                  "N must be a power of 2 and lies between 1 ~ {max_N}");')
    lines.append('')

    n = 1
    first = True
    while n <= max_N:
        cond = 'if constexpr' if first else '} else if constexpr'
        first = False
        lines.append(f'    {cond} (N == {n}) {{')
        lines.append(gen_asm_block(shape_str, n, regs_per_x, pack_str))
        n *= 2
    lines.append('    } else {')
    lines.append('      asm volatile("trap");')
    lines.append('    }')
    lines.append('  }')
    lines.append('};')
    return '\n'.join(lines)


out = []
out.append('#pragma once')
out.append('')
out.append('#include <cstdint>')
out.append('#ifndef __CUDACC_RTC__')
out.append('#include <cuda.h>')
out.append('#endif')
out.append('')
out.append('#include "common.h"')
out.append('')
out.append('namespace tl {')
out.append('')

for shape_str, class_name, regs_per_x, max_N in shapes:
    dp = shape_str.split('x')[0]
    bits = shape_str.split('x')[1]
    out.append(f'// {dp} data path lanes, {bits}-bit pattern, repeated N times (store)')
    out.append(f'template <bool Unpack16> class {class_name};')
    out.append('')
    # Non-unpack variant
    out.append(gen_class(shape_str, class_name, regs_per_x, max_N, False))
    # Unpack variant
    out.append(gen_class(shape_str, class_name, regs_per_x, max_N, True))
    out.append('')

# Composite 32dp classes
composites = [
    ('32dp64bNx', '16dp64bNx', 1),
    ('32dp128bNx', '16dp128bNx', 2),
    ('32dp256bNx', '16dp256bNx', 4),
]
for comp_suffix, base_suffix, mult in composites:
    out.append(f'// 32 data path lanes, composite via 2x16dp (store)')
    out.append(f'template <bool Unpack16 = false> class tmem_st_{comp_suffix} {{')
    out.append('public:')
    out.append('  template <int N>')
    out.append(f'  static TL_DEVICE void copy(uint32_t const &dst_addr, uint32_t const *src_ptr) {{')
    out.append(f'    tmem_st_{base_suffix}<Unpack16>::template copy<N>(dst_addr, src_ptr);')
    out.append(f'    tmem_st_{base_suffix}<Unpack16>::template copy<N>(dst_addr + (16 << 16), src_ptr + N{"" if mult == 1 else " * " + str(mult)});')
    out.append('  }')
    out.append('};')
    out.append('')

out.append('} // namespace tl')
out.append('')

content = '\n'.join(out)
path = 'src/tl_templates/cuda/tcgen_05_st.h'
with open(path, 'w') as f:
    f.write(content)
print(f'Written {len(content)} bytes to {path}')
print(f'Total lines: {len(out)}')
