#!/bin/bash

VERSION=v0.1
CDIR=`pwd`

INSTALLDIR=/usr/local/lumos
BINDIR=/usr/local/bin/lumos

if [ -d "$INSTALLDIR" ]; then
    rm -rf "$INSTALLDIR"
fi

if [ -d "$BINDIR" ]; then
    rm -rf "$BINDIR"
fi

ARCHIVE=`awk '/^__ARCHIVE_BOUNDARY__/ { print NR + 1; exit 0; }' $0`

tail -n +$ARCHIVE $0 > lumos.tar.gz
tar -zpxf lumos.tar.gz
wait

mv lumos-$VERSION $INSTALLDIR
cp $INSTALLDIR/bin/lumos $BINDIR

rm lumos.tar.gz

exit 0
__ARCHIVE_BOUNDARY__
