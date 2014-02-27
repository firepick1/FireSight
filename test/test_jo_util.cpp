#include <string.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include "FireLog.h"
#include "FireSight.hpp"
#include "jansson.h"
#include "jo_util.hpp"

using namespace std;
using namespace firesight;

void test_jo_parse() {
	map<string, const char *> args;
	args["red"] = "tomato";
	args["blue"] = "sky";
	string result;

	result = jo_parse("the {{red}} {{red}} sauce", args);
	cout << result << endl;
	assert(strcmp("the tomato tomato sauce", result.c_str()) == 0);
	result = jo_parse("Look at the {{blue}}.", args);
	cout << result << endl;
	assert(strcmp("Look at the sky.", result.c_str()) == 0);
}

void test_jo_util() {
	test_jo_parse();
}
