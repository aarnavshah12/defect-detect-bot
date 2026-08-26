#!/bin/sh
# Builds uvc-util (https://github.com/jtfrey/uvc-util, BSD licence) with the Xcode clang. Run from anywhere.
cd "$(dirname "$0")" && clang -fno-objc-arc -O2 -Wno-everything -framework Foundation -framework IOKit -framework CoreFoundation src/*.m -o uvc-util && echo "built $(pwd)/uvc-util"
