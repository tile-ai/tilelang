#!/usr/bin/env bash

echo "Add MIT license boilerplate..."
PWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# TO source code root
pushd "${PWD}/../../" > /dev/null

EXITCODE=0

for SRC_FILE in $(find . -path './3rdparty' -prune -false -o -path './build' -prune -false -o -type f -not -name \
    '*apply_mit_liscense.sh' -not -name '*check_mit_liscense.sh' -and \( -name '*.cpp' -or -name '*.h*' -or -name '*.cu' -or -name '*.in' \) ); do
    sed -i '/\/\/\s*Microsoft\s*(c)/Id' ${SRC_FILE}
    if !(grep -q "Copyright (c) Tile-AI Corporation." "${SRC_FILE}"); then
        cat maint/scripts/mit_liscense1.txt ${SRC_FILE} > ${SRC_FILE}.new
        mv ${SRC_FILE}.new ${SRC_FILE}
    fi
done

for SRC_FILE in $(find . -path './3rdparty' -prune -false -o -path './build' -prune -false -o -type f -not -name \
    '*apply_mit_liscense.sh' -not -name '*check_mit_liscense.sh' -and \( -name 'CMakeLists.txt' -or -name '*.cmake' \
    -or -name '*.py' -or -name '*.dockerfile' -or -name '*.yaml' \) ); do
    sed -i '/\#\s*Microsoft\s*(c)/Id' ${SRC_FILE}
    if !(grep -q "Copyright (c) Tile-AI Corporation." "${SRC_FILE}"); then
        cat maint/scripts/mit_liscense2.txt ${SRC_FILE} > ${SRC_FILE}.new
        mv ${SRC_FILE}.new ${SRC_FILE}
    fi
done

for SRC_FILE in $(find . -path './3rdparty' -prune -false -o -path './build' -prune -false -o -type f -not -name \
    '*apply_mit_liscense.sh' -not -name '*check_mit_liscense.sh' -name '*.sh' ); do
    sed -i '/\#\s*Microsoft\s*(c)/Id' ${SRC_FILE}
    if !(grep -q "Copyright (c) Tile-AI Corporation." "${SRC_FILE}"); then
        line=$(head -n 1 ${SRC_FILE})
        if [[ $line == "#!/bin/bash"* ]]; then
            (echo ${line}; echo ''; cat maint/scripts/mit_liscense2.txt; echo "$(tail -n +2 "${SRC_FILE}")" ) > ${SRC_FILE}.new
        else
            cat maint/scripts/mit_liscense2.txt ${SRC_FILE} > ${SRC_FILE}.new
        fi
        mv ${SRC_FILE}.new ${SRC_FILE}
    fi
done

echo "Done."
popd > /dev/null
exit $EXITCODE
