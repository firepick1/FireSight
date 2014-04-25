#! /bin/bash

fname=$1
if [ "$fname" == "" ]; then 
  fname="target/w-morph" 
fi
echo "generating images: $fname..."
target/firesight -i img/w.png -p json/morph.json -Dksize=3 -Dmop=MORPH_DILATE -ji 0 -o "$fname-dilate-3.png" $2
target/firesight -i img/w.png -p json/morph.json -Dksize=5 -Dmop=MORPH_DILATE -ji 0 -o "$fname-dilate-5.png" $2
target/firesight -i img/w.png -p json/morph.json -Dksize=3 -Dmop=MORPH_ERODE -ji 0 -o "$fname-erode-3.png" $2
target/firesight -i img/w.png -p json/morph.json -Dksize=5 -Dmop=MORPH_ERODE -ji 0 -o "$fname-erode-5.png" $2
target/firesight -i img/w.png -p json/morph.json -Dksize=3 -Dmop=MORPH_OPEN -ji 0 -o "$fname-open-3.png" $2
target/firesight -i img/w.png -p json/morph.json -Dksize=5 -Dmop=MORPH_OPEN -ji 0 -o "$fname-open-5.png" $2
target/firesight -i img/w.png -p json/morph.json -Dksize=3 -Dmop=MORPH_CLOSE -ji 0 -o "$fname-close-3.png" $2
target/firesight -i img/w.png -p json/morph.json -Dksize=5 -Dmop=MORPH_CLOSE -ji 0 -o "$fname-close-5.png" $2
target/firesight -i img/w.png -p json/morph.json -Dksize=3 -Dmop=MORPH_GRADIENT -ji 0 -o "$fname-gradient-3.png" $2
target/firesight -i img/w.png -p json/morph.json -Dksize=5 -Dmop=MORPH_GRADIENT -ji 0 -o "$fname-gradient-5.png" $2
target/firesight -i img/w.png -p json/morph.json -Dksize=3 -Dmop=MORPH_TOPHAT -ji 0 -o "$fname-tophat-3.png" $2
target/firesight -i img/w.png -p json/morph.json -Dksize=5 -Dmop=MORPH_TOPHAT -ji 0 -o "$fname-tophat-5.png" $2
target/firesight -i img/w.png -p json/morph.json -Dksize=3 -Dmop=MORPH_BLACKHAT -ji 0 -o "$fname-blackhat-3.png" $2
target/firesight -i img/w.png -p json/morph.json -Dksize=5 -Dmop=MORPH_BLACKHAT -ji 0 -o "$fname-blackhat-5.png" $2
