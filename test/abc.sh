#! /bin/bash

imgOut=$1
if [ "$imgOut" == "" ]; then
  echo "Generating target/abc.png"
  imgOut=target/abc.png
fi
bangColor=$2
if [ "$bangColor" == "" ]; then
  echo "Exclamation color is [255,0,255]"
  bangColor=[255,0,255]
fi
  
target/firesight -p json/rectangle.json -o $imgOut -Dwidth=200 -Dheight=50 -Dcolor=[0,0,0] -Dfill=[34,34,34]
target/firesight -i $imgOut -p json/putText.json -o $imgOut -Dorg=[10,-10] -Dtext=A -Dcolor=[20,20,20] -DfontScale=2 -Ditalic=true -Dthickness=2
target/firesight -i $imgOut -p json/putText.json -o $imgOut -Dorg=[30,-10] -Dtext=B -Dcolor=[24,24,24] -DfontScale=2 -Ditalic=true -Dthickness=2
target/firesight -i $imgOut -p json/putText.json -o $imgOut -Dorg=[50,-10] -Dtext=C -Dcolor=[28,28,28] -DfontScale=2 -Ditalic=true -Dthickness=2
target/firesight -i $imgOut -p json/putText.json -o $imgOut -Dorg=[70,-10] -Dtext=D -Dcolor=[32,32,32] -DfontScale=2 -Ditalic=true -Dthickness=2
target/firesight -i $imgOut -p json/putText.json -o $imgOut -Dorg=[90,-10] -Dtext=E -Dcolor=[36,36,36] -DfontScale=2 -Ditalic=true -Dthickness=2
target/firesight -i $imgOut -p json/putText.json -o $imgOut -Dorg=[110,-10] -Dtext=F -Dcolor=[40,40,40] -DfontScale=2 -Ditalic=true -Dthickness=2
target/firesight -i $imgOut -p json/putText.json -o $imgOut -Dorg=[130,-10] -Dtext=G -Dcolor=[44,44,44] -DfontScale=2 -Ditalic=true -Dthickness=2
target/firesight -i $imgOut -p json/putText.json -o $imgOut -Dorg=[150,-10] -Dtext=H -Dcolor=[48,48,48] -DfontScale=2 -Ditalic=true -Dthickness=2
target/firesight -i $imgOut -p json/putText.json -o $imgOut -Dorg=[170,-10] -Dtext=I -Dcolor=[52,52,52] -DfontScale=2 -Ditalic=true -Dthickness=2
target/firesight -i $imgOut -p json/putText.json -o $imgOut -Dorg=[185,-10] -Dtext=! -Dcolor=$bangColor -DfontScale=2 -Ditalic=true -Dthickness=2
