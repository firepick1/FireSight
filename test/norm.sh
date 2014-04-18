#! /bin/bash

function help() {
  echo help
}

args=
domain=
outdir=~/Downloads
prefix=kodak_test
y=-10

while getopts "a:o:p:d:" flag
do
  case "$flag" in
    a) args="$OPTARG";;
    d) domain="$OPTARG";;
    p) prefix="$OPTARG";;
    o) outdir="$OPTARG";;
    *) help ; exit 0;;
  esac
done

echo "  Directory  : $outdir"
echo "  Prefix     : $prefix"
echo "  Domain     : $domain"

image=$outdir/$prefix.png

target/firesight -i $image -p json/normalize.json -o $outdir/$prefix-inf.png -DnormType=NORM_INF -ji 0 $args 
target/firesight -i $image -p json/normalize.json -o $outdir/$prefix-l1.png -DnormType=NORM_L1   -ji 0 $args
target/firesight -i $image -p json/normalize.json -o $outdir/$prefix-l2.png -DnormType=NORM_L2   -ji 0 $args
target/firesight -i $image -p json/normalize.json -o $outdir/$prefix-minmax.png -DnormType=NORM_MINMAX -ji 0  $args
target/firesight -i $image -p json/normalize.json -o $outdir/$prefix-inf-dom.png -DnormType=NORM_INF -ji 0 -Ddomain=$domain $args
target/firesight -i $image -p json/normalize.json -o $outdir/$prefix-l1-dom.png -DnormType=NORM_L1  -ji 0 -Ddomain=$domain $args
target/firesight -i $image -p json/normalize.json -o $outdir/$prefix-l2-dom.png -DnormType=NORM_L2  -ji 0 -Ddomain=$domain $args
target/firesight -i $image -p json/normalize.json -o $outdir/$prefix-minmax-dom.png -DnormType=NORM_MINMAX  -ji 0 -Ddomain=$domain $args
