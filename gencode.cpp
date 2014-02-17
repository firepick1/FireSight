#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>

using namespace std;

void help() {
	cout << "Generate FireSight code:" << endl;
	cout << "  gencode -ringMat 128" << endl;
}

void generate_ringMat(short radius) {
	cout << "short ringMap[" << radius << "][" << radius << "] = {" << endl;
	for (int r=0; r<radius; r++) {
		cout << "/*" << r << "*/";
		for (int c=0; c<radius; c++) {
			int d = floor(sqrt(r*r+c*c));
			cout << ((r || c) ? "," : " ");
			cout << d;
		}
		cout << endl;
	}
	cout << "};" << endl;
}

int main(int argc, char *argv[]) {
	int ringMat_radius = 128;
	for (int i = 1; i < argc; i++) {
		if (strcmp("-ringMat",argv[i]) == 0) {
			if (i+1>=argc) {
				cout << "ERROR: expected -ringMat SIZE (default 128)" << endl;
				exit(-1);
			}
			i++;
			sscanf(argv[i], "%d", &ringMat_radius);
		} else {
			help();
			exit(-1);
		}
	}

	cout << "///////// BEGIN GENERATED CODE" << endl;
	cout << "namespace FireSight {" << endl;
	generate_ringMat(ringMat_radius);
	cout << "} //namespace FireSight" << endl;
	cout << "///////// END GENERATED CODE" << endl;
}
