#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
"""
Review workspace creation tool

Called by the plan-design / clause-grouping subagent (router) during Stage 0.
Creates a structured YAML output directory under /tmp for the current review,
where the collector stores YAML results submitted by clause-review subagents.

Directory naming rules:
  - PR review:    pr{PR number}_{random token}    for example, pr1234_a3b7x9
  - File review:  file_{random token}             for example, file_k8m2q3

Usage:
  python3 workflow.create_review_dir.py --type pr --id 1234
  python3 workflow.create_review_dir.py --type file

Output: Prints the absolute directory path to stdout
Exit codes: 0=success, 1=invalid arguments, 2=directory creation failure
"""

import argparse
import logging
import os
import random
import string
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)


def gen_random_token(length: int = 6) -> str:
    """Generate a random token containing lowercase letters and digits."""
    alphabet = string.ascii_lowercase + string.digits
    return "".join(random.choices(alphabet, k=length))


def build_dir_name(review_type: str, review_id: str) -> str:
    """Construct a directory name according to the naming rules."""
    token = gen_random_token()
    if review_type == "pr":
        if not review_id:
            raise ValueError("PR reviews require --id (the PR number)")
        return f"pr{review_id}_{token}"
    if review_type == "file":
        return f"file_{token}"
    raise ValueError(f"Unknown review type: {review_type} (only pr / file are supported)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a review YAML output directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--type",
        required=True,
        choices=["pr", "file"],
        help="Review type: pr=PR review, file=file review",
    )
    parser.add_argument(
        "--id",
        default="",
        help="Review identifier: PR number for a PR review; leave empty for a file review",
    )
    args = parser.parse_args()

    try:
        dir_name = build_dir_name(args.type, args.id)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    dir_path = os.path.join("/tmp", dir_name)

    try:
        os.makedirs(dir_path, mode=0o755, exist_ok=False)
    except FileExistsError:
        # Retry once in the unlikely event of a collision.
        dir_name = build_dir_name(args.type, args.id)
        dir_path = os.path.join("/tmp", dir_name)
        try:
            os.makedirs(dir_path, mode=0o755, exist_ok=False)
        except Exception as exc:
            logger.error("Failed to create directory: %s", exc)
            return 2
    except Exception as exc:
        logger.error("Failed to create directory: %s", exc)
        return 2

    # Print only the absolute directory path to stdout so the router subagent can capture it and return it to the main agent.
    print(dir_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
