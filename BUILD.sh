#!/bin/bash

CURDIR=`pwd`
BUILDDIR=$CURDIR/lumos-build

if [ -d "$BUILDDIR" ]; then
    rm -rf "$BUILDDIR"
fi

mkdir lumos-build
cp -r ./include ./lumos-build/include
cp -r ./lumos/lib ./lumos-build/lib
cp -r ./lumos ./lumos-build/src
cp -r ./Lumos-BUILD ./lumos-build/makefile
cp -r ./data ./lumos-build/data

tar zcvf lumos.tar.gz ./lumos-build
cat install.sh lumos.tar.gz > lumos.run

rm -rf $BUILDDIR
rm -f lumos.tar.gz
