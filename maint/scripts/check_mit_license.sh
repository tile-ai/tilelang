#!/usr/bin/env bash

echo "Check MIT License boilerplate..."
PWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# To source code root
pushd "${PWD}/../../" > /dev/null

EXITCODE=0

for SRC_FILE in $(find . -path './3rdparty' -prune -false -o -path './build' -prune -false -o -type f -not -name '*apply_mit_license.sh' \
    -not -name '*check_mit_license.sh' -and \( -name 'CMakeLists.txt' -or -name '*.cpp' -or -name '*.cu' -or -name '*.h'  -or -name '*.hpp' \
    -or -name '*.py' -or -name '*.sh' -or -name '*.dockerfile' -or -name '*.yaml' \) ); do

    # Skip files that already contain the Apache License
    if grep -q "Apache License" "${SRC_FILE}"; then
        continue
    fi

    if !(grep -q "Copyright (c) Tile-AI Corporation." "${SRC_FILE}") || !(grep -q "Licensed under the MIT License." "${SRC_FILE}") \
    || (grep -q -i -P "Microsoft( |)\(c\)" "${SRC_FILE}"); then
        echo "[ERROR] Require: MIT License boilerplate" "${SRC_FILE}"
        EXITCODE=1
    fi
done

echo "Done."
popd > /dev/null
exit $EXITCODE
