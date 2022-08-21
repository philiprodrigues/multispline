#pragma once

#include <immintrin.h>

const int N = 7; // number of knots used for each spline
// x values of the knots
double xs[N] = {  -3,  -2,   -1, 0,    1,   2,   3};

// Method 1 of saving the splines. Could call this "structure-of-arrays"
struct MySpline
{
  double ys[N];
  double bs[N];
  double cs[N];
  double ds[N];
};

struct MySpline_float
{
  float ys[N];
  float bs[N];
  float cs[N];
  float ds[N];
};

// Method 2 of saving the splines. Could call this "array-of-structures"
struct Coeffs {
  double y, b, c, d;
};
struct Coeffs_float {
  float y, b, c, d;
};

struct MySplineAoS
{
  Coeffs c[N];
};

struct MySplineAoS_float
{
  Coeffs_float c[N];
};

// Method 3 of saving the splines. We save 4 splines' coefficients in
// each instance (because 4 is the number of doubles that fit in an
// AVX2 register)
struct MySplineAVX
{
  __m256d ys[N];
  __m256d bs[N];
  __m256d cs[N];
  __m256d ds[N];
};
