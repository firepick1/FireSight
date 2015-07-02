/*
 * @Author  : Simon Fojtu (simon.fojtu@gmail.com)
 * @Date    : 10.07.2014
 */

#include <vector>
#include <math.h>
#include <stdexcept>
#include "FireLog.h"
#include "jansson.h"
#include "points2resolution.h"

using namespace std;
namespace firesight {

bool Pt2Res::compare_XY_by_x(XY a, XY b) {
    return a.x < b.x;
}

bool Pt2Res::compare_XY_by_y(XY a, XY b) {
    return a.y < b.y;
}

int Pt2Res::nsamples_RANSAC(size_t ninl, size_t xlen, unsigned int NSAMPL, double confidence) {
    // q = \prod_{i=0}^{NSAMPL-1} (ninl-i)/(xlen-i)
    double q = 1;
    int nsamples;
    for (unsigned int i = 0; i < NSAMPL - 1; i++)
        q *= (double) (ninl-i)/(xlen-i);

    if (q < 1e-10)
        return INT_MAX;

     nsamples = (int) log(1-confidence) / log(1-q);
     if (nsamples < 1)
         nsamples = 1;

     return nsamples;
}

double Pt2Res::_RANSAC_line(XY * x, size_t nx, XY C) {
    assert(nx == 2);

    XY u(x[1].x - x[0].x, x[1].y - x[0].y);
    double norm_u = sqrt(u.x*u.x + u.y*u.y);
    XY p(x[1].x - C.x, x[1].y - C.y);
    double e = abs(u.x*p.y - u.y*p.x) / norm_u;

    return e;
}

double Pt2Res::_RANSAC_pattern(XY * x, size_t nx, XY C) {
    assert(nx == 2);

    XY u(x[1].x - x[0].x, x[1].y - x[0].y);
    // squared distance from x[0] to x[1]
    double usq = u.x*u.x + u.y*u.y;
    // vector from x[0] to C determines the unit vector
    XY Ap(C.x - x[0].x, C.y - x[0].y);
    // projection of C onto the line parametrized as x[0] + t (x[1] - x[0])
    double t = (Ap.x * u.x + Ap.y * u.y) / usq;

    return abs(t-round(t));
}

vector<XY> Pt2Res::RANSAC_2D(unsigned int NSAMPL, vector<XY> coords, double thr, double confidence, double(*err_fun)(XY *, size_t, XY)) {

    double bsupp = -1;
    vector<double> berr;
    vector<XY> binl;
    unsigned int max_iterations = 1000;
    if (coords.size() < NSAMPL)
        return binl; //errMsg = "Not enough detected circles, at least 2 needed";

    for (unsigned int it = 0; it < max_iterations; it++) {
        int idx0, idx1;
        vector<double> err;
        vector<XY> inl;
        double supp = 0;

        XY * sampl = (XY *) malloc(sizeof(XY) * NSAMPL);
        int * sampl_id = (int *) malloc(sizeof(int) * NSAMPL);
        for (unsigned int ni = 0; ni < NSAMPL; ni++) {
            bool rerun;
            do {
                rerun = false;
                sampl_id[ni] = rand() % coords.size();
                for (unsigned int oi = 0; oi < ni; oi++) {
                    if (sampl_id[ni] == sampl_id[oi]) {
                        rerun = true;
                        break;
                    }
                }
            } while (rerun);
            sampl[ni] = coords[sampl_id[ni]];
        }


        for (size_t cid = 0; cid < coords.size(); cid++) {
            double e;

            XY C = coords[cid];

            e = err_fun(sampl, NSAMPL, C);

            err.push_back(e);

            if (e < thr) {
                inl.push_back(C);
                supp += (1-e*e);
            }
        }
        free(sampl);
        free(sampl_id);

        // support - approximation of ML estimator
        supp /= thr * thr * coords.size();

        if (supp > bsupp) {
            bsupp = supp;
            binl = inl;
            berr = err;

            // update max_iterations
            max_iterations = nsamples_RANSAC(inl.size(), coords.size(), NSAMPL, confidence);
        }
    }
}

void Pt2Res::least_squares(vector<XY> xy, double * a, double * b) {
    double SUMx = 0, SUMy = 0, SUMxy = 0, SUMxx = 0;

    for (size_t i = 0; i < xy.size(); i++) {
        SUMx = SUMx + xy[i].x;
        SUMy = SUMy + xy[i].y;
        SUMxy = SUMxy + xy[i].x*xy[i].y;
        SUMxx = SUMxx + xy[i].x*xy[i].x;
    }
    *a = ( SUMx*SUMy - xy.size()*SUMxy ) / ( SUMx*SUMx - xy.size()*SUMxx );
    *b = ( SUMy - (*a)*SUMx ) / xy.size();
}

double Pt2Res::getResolution(double thr1, double thr2, double confidence, double separation, vector<XY> coords) {
    double resolution = NAN;

    try {
        if (coords.size() < 2) {
            throw runtime_error("Not enough points given");
        }

        // fit line through circle centers
        vector<XY> binl = RANSAC_2D(2, coords, thr1, confidence, _RANSAC_line);

        if (binl.size() < 2) {
            throw runtime_error("Not enough points after RANSAC line");
        }

        // fit a line through the inliers
        double a, b;
        least_squares(binl, &a, &b);

        // run another RANSAC to get the pattern in the circle centers forming the line
        binl = RANSAC_2D(2, binl, thr2, confidence, _RANSAC_pattern);

        if (binl.size() < 2) {
            throw runtime_error("Not enough points after RANSAC pattern");
        }

        // sort the inliers
        if (abs(a) > 1)
            sort(binl.begin(), binl.end(), compare_XY_by_y);
        else
            sort(binl.begin(), binl.end(), compare_XY_by_x);

        // compute distance of neighbours (inter_d)
        vector<double> inter_d;
        for (size_t i = 1; i < binl.size(); i++)
            inter_d.push_back(sqrt(
                        (binl[i].x - binl[i-1].x)*(binl[i].x - binl[i-1].x) + 
                        (binl[i].y - binl[i-1].y)*(binl[i].y - binl[i-1].y)));

        // get the median (d0) of the distances
        sort(inter_d.begin(), inter_d.end());
        double d0;
        if (inter_d.size() % 2 == 0)
            d0 = (inter_d[inter_d.size() / 2] + inter_d[inter_d.size() / 2 - 1])/2;
        else
            d0 = inter_d[inter_d.size() / 2];

        resolution = d0 / separation;

    } catch (exception &e) {
        LOGERROR(e.what());
    }

    return resolution;
}

} // namespace
