"""Archived 1SM non-split experiment.

The maintained 1SM SM100 target is ``attention_kernel_1sm.py``.  This
legacy file used monolithic role helpers that are no longer public TileLang
primitives.
"""

from __future__ import annotations

import argparse


def attention_kernel_1sm(*args, **kwargs):
    raise RuntimeError(
        "attention_kernel_1sm_nosplit.py is archived. Use "
        "examples/flash_attention_sm100/attention_kernel_1sm.py for the "
        "maintained DSL split path."
    )


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    attention_kernel_1sm()


if __name__ == "__main__":
    main()
