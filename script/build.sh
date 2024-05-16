#!/bin/bash
VERSION=v0.1
CDIR=`pwd`

BUILDDIR=$CDIR/build

if [ -d "$BUILDDIR" ]; then
    rm -rf "$BUILDDIR"
fi

mkdir $BUILDDIR
mkdir $BUILDDIR/include
mkdir $BUILDDIR/lib
mkdir $BUILDDIR/obj
wait

# cp $CDIR/include/lumos.h $BUILDDIR/include/lumos.h
# wait

# tar zcvf lumos-$VERSION.tar.gz lumos-$VERSION
# wait

# if [ -d "$BUILDDIR" ]; then
#     rm -rf "$BUILDDIR"
# fi

# cat LumosInstall.sh lumos-$VERSION.tar.gz > lumos-$VERSION.run
# rm lumos-$VERSION.tar.gz
