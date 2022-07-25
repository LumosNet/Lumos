#!/bin/bash

CDIR=`pwd`

BUILDDIR=$CDIR/lumos-build
INCLUDEDIR=$CDIR/lumos-include
LIBDIR=$CDIR/lumos-lib
SRCDIR=$CDIR/lumos-src
INSTALLDIR=/usr/local/lumos
WORKSPACE=/home/lumos

if [ -d "$BUILDDIR" ]; then
    rm -rf "$BUILDDIR"
fi

if [ -d "$INCLUDEDIR" ]; then
    rm -rf "$INCLUDEDIR"
fi

if [ -d "$LIBDIR" ]; then
    rm -rf "$LIBDIR"
fi

if [ -d "$SRCDIR" ]; then
    rm -rf "$SRCDIR"
fi

if [ -d "$INSTALLDIR" ]; then
    rm -rf "$INSTALLDIR"
fi

if [ -d "$WORKSPACE" ]; then
    rm -rf "$WORKSPACE"
fi

ARCHIVE=`awk '/^__ARCHIVE_BOUNDARY__/ { print NR + 1; exit 0; }' $0`

tail -n +$ARCHIVE $0 > lumos.tar.gz
tar -zpxf lumos.tar.gz
wait

rm lumos.tar.gz

mv ./lumos-build/include ./lumos-include
mv ./lumos-build/lib ./lumos-lib
mv ./lumos-build/src ./lumos-src
mv ./lumos-build/makefile ./lumos-makefile

mkdir ./lumos-build/lib
mkdir ./lumos-build/bin
mkdir ./lumos-build/include

make -f lumos-makefile
wait

mv ./lumos-include/lumos.h ./lumos-build/include/lumos.h

mkdir /usr/local/lumos
cp -r ./lumos-build/bin /usr/local/lumos/bin
cp -r ./lumos-src/lumos/demos /usr/local/lumos/bin/demos
cp -r ./lumos-build/include /usr/local/lumos/include
cp -r ./lumos-build/lib /usr/local/lumos/lib
cp /usr/local/lumos/bin/lumos /usr/local/bin/lumos

rm -rf ./lumos-build
rm -rf ./lumos-include
rm -rf ./lumos-lib
rm -rf ./lumos-src
rm -rf ./lumos-obj
rm -f ./lumos-makefile

mkdir $WORKSPACE

exit 0
__ARCHIVE_BOUNDARY__
