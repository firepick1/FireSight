#ifndef MAPPARSER_H
#define MAPPARSER_H

#include <map>
#include <string>
#include <algorithm>
#include <stdexcept>

using namespace std;

namespace firesight {

template< class SpecParser >
class MapParser {
public:
    static map<int, string> get() {
        return SpecParser::get();
    }

    static string get(int key) {
        map<int, string> amap = SpecParser::get();
        return amap[key];
    }

    static int get(string value) {
        map<int, string> amap = SpecParser::get();
        int key;

        auto findType = std::find_if(std::begin(amap), std::end(amap), [&](const std::pair<int, string> &pair)
        {
            return value.compare(pair.second) == 0;
        });
        if (findType != std::end(amap)) {
            key = findType->first;
        } else
            throw std::invalid_argument("unknown value '" + value + "'");

        return key;
    }
};



class CvTypeParser_ : public MapParser< CvTypeParser_ > {
    friend class MapParser< CvTypeParser_ >;
public:
    static map<int, string> get() { return amap; }

private:
    static std::map<int, string> amap;
};

typedef MapParser<CvTypeParser_> CvTypeParser;



class BorderTypeParser_ : public MapParser< BorderTypeParser_ > {
    friend class MapParser< BorderTypeParser_ >;
public:
    static map<int, string> get() { return amap; }

private:
    static std::map<int, string> amap;
};

typedef MapParser<BorderTypeParser_> BorderTypeParser;



enum BlurType {
    BILATERAL,
    BILATERAL_ADAPTIVE,
    BOX,
    BOX_NORMALIZED,
    GAUSSIAN,
    MEDIAN
};

class BlurTypeParser_ : public MapParser< BlurTypeParser_ > {
    friend class MapParser< BlurTypeParser_ >;
public:
    static map<int, string> get() { return amap; }

private:
    static std::map<int, string> amap;
};

typedef MapParser<BlurTypeParser_> BlurTypeParser;


enum BGSubtractionType {
    MOG,
    MOG2,
    ABSDIFF
};

class BGSubTypeParser_ : public MapParser< BGSubTypeParser_ > {
    friend class MapParser< BGSubTypeParser_ >;
public:
    static map<int, string> get() { return amap; }

private:
    static std::map<int, string> amap;
};

typedef MapParser<BGSubTypeParser_> BGSubTypeParser;

} // namespace

#endif // MAPPARSER_H
