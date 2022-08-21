#include "multispline.h"

#include "TSpline.h"
#include "TRandom3.h"

#include <emmintrin.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>

#include <immintrin.h>

inline double eval_spline(const MySpline& s, const double x)
{
  int which = std::floor(x + 3);
  const double xknot = -3 + which;
  const double dx = x - xknot;
  return (s.ys[which]+dx*(s.bs[which]+dx*(s.cs[which]+dx*s.ds[which]))); 
}

inline double eval_splineAoS(const MySplineAoS& s, const double x)
{
  int which = std::floor(x + 3);
  const double xknot = -3 + which;
  const double dx = x - xknot;
  const Coeffs& c=s.c[which];
  return (c.y + dx*(c.b + dx*(c.c + dx*c.d))); 
}

// Evaluate the splines stored in the MySplineAVX at the same value `x` and store all the outputs in `weights`
inline void eval_splineAVX(const MySplineAVX& s, double x, double* weights)
{
  int which = std::floor(x + 3);
  const double xknot = -3 + which;
  // Set all of the items in the dx register to the same value
  __m256d dx = _mm256_set1_pd(x - xknot);

  // Choose the coefficients for the relevant knot (the same for all splines)
  __m256d y = s.ys[which];
  __m256d b = s.bs[which];
  __m256d c = s.cs[which];
  __m256d d = s.ds[which];

  // Calculate the weight (y + dx*(b + dx*(c + dx*d)))
  //
  // _m256_add_pd adds the two registers elementwise
  // _m256_mul_pd multiplies the two registers elementwise
  __m256d weight = _mm256_add_pd(c, _mm256_mul_pd(dx, d));
  weight = _mm256_add_pd(b, _mm256_mul_pd(dx, weight));
  weight = _mm256_add_pd(y, _mm256_mul_pd(dx, weight));

  // Store the output to `weights` (the 'u' in 'storeu' is for
  // 'unaligned', ie the output pointer is not necessarily aligned to
  // a 256-bit boundary)
  _mm256_storeu_pd(weights, weight);
}

MySplineAoS make_splineAoS(const MySpline& source)
{
  MySplineAoS s;
  for (int i=0; i<N; ++i) {
    s.c[i].y=source.ys[i];
    s.c[i].b=source.bs[i];
    s.c[i].c=source.cs[i];
    s.c[i].d=source.ds[i];
  }

  return s;
}

// Make a MySplineAVX object from the 4 MySplines starting at `source`
MySplineAVX make_splineAVX(const MySpline* source)
{
  MySplineAVX s;
  for (int i=0; i<N; ++i) {
    s.ys[i] = _mm256_setr_pd(source[0].ys[i], source[1].ys[i], source[2].ys[i], source[3].ys[i]);
    s.bs[i] = _mm256_setr_pd(source[0].bs[i], source[1].bs[i], source[2].bs[i], source[3].bs[i]);
    s.cs[i] = _mm256_setr_pd(source[0].cs[i], source[1].cs[i], source[2].cs[i], source[3].cs[i]);
    s.ds[i] = _mm256_setr_pd(source[0].ds[i], source[1].ds[i], source[2].ds[i], source[3].ds[i]);
  }

  return s;
}

// The current time in microseconds
uint64_t now_us()
{
  using namespace std::chrono;
  return duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
}

void benchmark(std::function<void(void)> fn)
{
  // We run many times because there is a large run-to-run variation
  const int nruns = 50;
  double best = std::numeric_limits<double>::max();
  double total = 0;
  for (int j=0; j<nruns; ++j) {
    uint64_t start = now_us();
    fn();
    // for (int i=0; i<nsplines; ++i) {
    //   weights[i] = eval_spline(splines[i], shifts[i]);
    // }
    uint64_t end = now_us();
    double dur = 1e-3*(end-start);
    best = std::min(best, dur);
    total += dur;
    std::cout << static_cast<int>(dur) << " ";
  }
  std::cout << std::endl;
  std::cout << "Mean: " << (total/nruns) << "ms, best: " << best << "ms" << std::endl;
}

bool check_weights(const std::vector<double> &weights,
                   const std::vector<double> &weights_gold)

{
  for (size_t i=0; i<weights.size(); ++i) {
    double gold = weights_gold[i];
    if (fabs(gold - weights[i]) > 1e-5) {
      return false;
    }
  }
  return true;
}

int main()
{
  const int nsplines = 10*1000*1000;

  // Spline objects for the 3 different methods
  std::vector<MySpline> splines(nsplines);
  std::vector<MySplineAoS> splinesAoS(nsplines);
  std::vector<MySplineAVX> splinesAVX(nsplines);

  // The systematic shifts
  std::vector<double> shifts(nsplines);

  // Weights from the three methods, and the "true" values from TSpline3
  std::vector<double> weights(nsplines);
  std::vector<double> weightsAoS(nsplines);
  std::vector<double> weightsAVX(nsplines);
  std::vector<double> weights_gold(nsplines);

  // Read in the inputs from the files created by make_random_inputs.cxx
  std::ifstream fin("spline_coeffs.dat", std::ios::binary);
  char* buffer = new char[nsplines*sizeof(MySpline)];
  fin.read(buffer, nsplines*sizeof(MySpline));

  std::ifstream fin_shifts("shifts.dat", std::ios::binary);
  char* shift_buffer = new char[nsplines*sizeof(double)];
  fin_shifts.read(shift_buffer, nsplines*sizeof(double));
  
  std::ifstream fin_weights("weights.dat", std::ios::binary);
  char* weight_buffer = new char[nsplines*sizeof(double)];
  fin_weights.read(weight_buffer, nsplines*sizeof(double));

  // Populate the spline objects, shifts and "true" weights
  std::cout << "Making random splines" << std::endl;
  for (int i=0; i<nsplines; ++i) {
    splines[i] = MySpline(*(reinterpret_cast<MySpline*>(buffer)+i));
    splinesAoS[i] = make_splineAoS(splines[i]);
    shifts[i] = *(reinterpret_cast<double*>(shift_buffer)+i);
    weights_gold[i] = *(reinterpret_cast<double*>(weight_buffer)+i);
  }

  for(int i=0; i<nsplines/4; ++i) {
    splinesAVX[i] = make_splineAVX(&splines[i*4]);
  }
  
  delete[] buffer;
  delete[] shift_buffer;
  delete[] weight_buffer;

  // =====================================================================
  //
  // Actually test the spline evaluations

  // -------------------------------------------------------------------
  std::cout << "Evaluating splines with MySpline" << std::endl;

  benchmark([&](){
    for (int i=0; i<nsplines; ++i) {
      weights[i] = eval_spline(splines[i], shifts[i]);
    }
  });

  // -------------------------------------------------------------------
  std::cout << "Evaluating splines with MySplineAoS" << std::endl;
  benchmark([&](){
    for (int i=0; i<nsplines; ++i) {
      weightsAoS[i] = eval_splineAoS(splinesAoS[i], shifts[i]);
    }
  });

  // -------------------------------------------------------------------
  std::cout << "Evaluating splines with eval_splineAVX" << std::endl;
  benchmark([&](){
    for (int i=0; i<nsplines/4; i++) {
      eval_splineAVX(splinesAVX[i], shifts[i*4], &weightsAVX[i*4]);
    }
  });

  // ===================================================================
  std::cout << "Checking against TSpline3..." << std::flush;

  bool good=check_weights(weights, weights_gold);
  bool goodAoS=check_weights(weightsAoS, weights_gold);
  bool goodAVX=check_weights(weightsAVX, weights_gold);

  std::cout << "MySpline: " << (good ? "good" : "bad") << ", MySplineAoS: " << (goodAoS ? "good" : "bad")
            << ", AVX: " << (goodAVX ? "good" : "bad") << std::endl;
  
}
