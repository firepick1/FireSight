#ifndef FIRESIGHTGUI_HPP
#define FIRESIGHTGUI_HPP

#include "opencv2/features2d/features2d.hpp"
#include <vector>
#include <map>
#ifdef _MSC_VER
#include "winjunk.hpp"
#else
#define CLASS_DECLSPEC
#endif
#include "jansson.h"
#include "Pipeline.h"

using namespace cv;
using namespace std;

#if __amd64__ || __x86_64__ || _WIN64 || _M_X64
#define FIRESIGHT_64_BIT
#define FIRESIGHT_PLATFORM_BITS 64
#else
#define FIRESIGHT_32_BIT
#define FIRESIGHT_PLATFORM_BITS 32
#endif

namespace firesight {



} // namespace firesight

#endif
