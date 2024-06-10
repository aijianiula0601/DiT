#!/bin/bash

set -ex

curdir=$(pwd)
echo "curdir:$curdir"
cd "$curdir" || exit

cd ../../
jupyter notebook --no-browser --allow-root --port 6114
