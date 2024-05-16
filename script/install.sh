#!/bin/bash

CDIR=`pwd`

INSTALLDIR=/usr/local/lumos
CUDAINCLUDE=/usr/local/cuda/include/
CUDALIB=/usr/local/cuda/lib/

if [ -d "$INSTALLDIR" ]; then
    rm -rf "$INSTALLDIR"
fi

ARCHIVE=`awk '/^__ARCHIVE_BOUNDARY__/ { print NR + 1; exit 0; }' $0`

tail -n +$ARCHIVE $0 > lumos.tar.gz
tar -zxvf lumos.tar.gz
wait

rm -f lumos.tar.gz

mv ./build/include $INSTALLDIR/include
mv ./build/lib ./lib
mv ./build/lumos ./lumos
mv ./build/makefile ./makefile

rm -rf ./build
mkdir ./build
mkdir ./build/lib
mkdir ./build/obj

make -f makefile CUDAINCLUDE=$CUDAINCLUDE CUDAINCLUDE=$CUDAINCLUDE
wait

mv ./build/lib $INSTALLDIR/lib

rm -rf build

exit 0
__ARCHIVE_BOUNDARY__
