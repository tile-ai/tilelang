# if dist and build directories exist, remove them
if [ -d dist ]; then
    rm -r dist
fi

if [ -d build ]; then
    rm -r build
fi

PYPI_BUILD=TRUE WITH_COMMITID=FALSE python setup.py bdist_wheel --plat-name=manylinux1_x86_64
