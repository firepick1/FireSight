FireSight
=========

FireSight is a C++ library for OpenCV image processing pipelines constructed from a JSON specification.
When processed, the pipeline transforms an optional input image and 
returns a JSON model with information recognized during each pipeline stage.

FireSight is available as a C++ library as well as a standalone runtime under the MIT license. It has been tested on:

* Raspbian (Raspberry Pi Debian)

### Installation

<code>
git clone git://github.com/firepick1/FireSight
cd FireSight
sudo ./build
</code>

### Examples
Recognize holes in cam.jpg:
<code>
target/firesight -p pipeline0.json -i cam.jpg
</code>



