#! /bin/bash

function help() {
  echo "abc options:"
  echo "  -b base intensity"
  echo "  -d domain"
  echo "  -f fill color [B,G,R]"
  echo "  -h height"
  echo "  -o output directory"
  echo "  -p file name prefix"
  echo "  -s text scale"
  echo "  -t text"
  echo "  -w width"
  echo "  -x exclamation color [B,G,R]"
  echo "  -y baseline position"

}

args=
domain=
outdir=target
prefix=abc
bang=[255,0,255]
fill=[34,34,34]
width=200
height=50
scale=2
thickness=2
base=20
y=-10

while getopts "a:o:p:d:x:f:w:h:s:b:y:a:" flag
do
  case "$flag" in
    a) args="$OPTARG";;
    b) base="$OPTARG";;
    d) domain="$OPTARG";;
    f) fill="$OPTARG";;
    h) height="$OPTARG";;
    p) prefix="$OPTARG";;
    o) outdir="$OPTARG";;
    s) scale="$OPTARG";;
    t) thickness="$OPTARG";;
    w) width="$OPTARG";;
    x) bang="$OPTARG";;
    y) y="$OPTARG";;

    *) help ; exit 0;;
  esac
done

abc=$outdir/$prefix.png

echo "Creating test image:"
echo "  Directory  : $outdir"
echo "  Exclamation: $bang"
echo "  Fill       : $fill"
echo "  Prefix     : $prefix"
echo "  Width      : $width"
echo "  Height     : $height"
echo "  Scale      : $scale"
echo "  Thickness  : $thickness"
echo "  Base       : $base"
echo "  Y          : $y"

textOpts="-DfontScale=$scale -Ditalic=true -Dthickness=$thickness -ji 0"
echo $textOpts

target/firesight -p json/rectangle.json -o $abc -Dwidth=$width -Dheight=$height -Dcolor=[0,0,0] -Dfill=$fill
target/firesight -i $abc -p json/putText.json -o $abc -Dorg=[10,$y] -Dtext=A -Dcolor=[$((base+0)),$((base+0)),$((base+0))] $textOpts
target/firesight -i $abc -p json/putText.json -o $abc -Dorg=[30,$y] -Dtext=B -Dcolor=[$((base+4)),$((base+4)),$((base+4))] $textOpts
target/firesight -i $abc -p json/putText.json -o $abc -Dorg=[50,$y] -Dtext=C -Dcolor=[$((base+8)),$((base+8)),$((base+8))] $textOpts
target/firesight -i $abc -p json/putText.json -o $abc -Dorg=[70,$y] -Dtext=D -Dcolor=[$((base+12)),$((base+12)),$((base+12))] $textOpts
target/firesight -i $abc -p json/putText.json -o $abc -Dorg=[90,$y] -Dtext=E -Dcolor=[$((base+16)),$((base+16)),$((base+16))] $textOpts
target/firesight -i $abc -p json/putText.json -o $abc -Dorg=[110,$y] -Dtext=F -Dcolor=[$((base+20)),$((base+20)),$((base+20))] $textOpts
target/firesight -i $abc -p json/putText.json -o $abc -Dorg=[130,$y] -Dtext=G -Dcolor=[$((base+24)),$((base+24)),$((base+24))] $textOpts
target/firesight -i $abc -p json/putText.json -o $abc -Dorg=[150,$y] -Dtext=H -Dcolor=[$((base+28)),$((base+28)),$((base+28))] $textOpts
target/firesight -i $abc -p json/putText.json -o $abc -Dorg=[170,$y] -Dtext=I -Dcolor=[$((base+32)),$((base+32)),$((base+32))] $textOpts
target/firesight -i $abc -p json/putText.json -o $abc -Dorg=[185,$y] -Dtext=! -Dcolor=$bang $textOpts

target/firesight -i $abc -p json/normalize.json -o $outdir/$prefix-inf.png -DnormType=NORM_INF -ji 0 $args 
target/firesight -i $abc -p json/normalize.json -o $outdir/$prefix-l1.png -DnormType=NORM_L1   -ji 0 $args
target/firesight -i $abc -p json/normalize.json -o $outdir/$prefix-l2.png -DnormType=NORM_L2   -ji 0 $args
target/firesight -i $abc -p json/normalize.json -o $outdir/$prefix-minmax.png -DnormType=NORM_MINMAX -ji 0  $args
target/firesight -i $abc -p json/normalize.json -o $outdir/$prefix-inf-dom.png -DnormType=NORM_INF -ji 0 -Ddomain=$domain $args
target/firesight -i $abc -p json/normalize.json -o $outdir/$prefix-l1-dom.png -DnormType=NORM_L1  -ji 0 -Ddomain=$domain $args
target/firesight -i $abc -p json/normalize.json -o $outdir/$prefix-l2-dom.png -DnormType=NORM_L2  -ji 0 -Ddomain=$domain $args
target/firesight -i $abc -p json/normalize.json -o $outdir/$prefix-minmax-dom.png -DnormType=NORM_MINMAX  -ji 0 -Ddomain=$domain $args
