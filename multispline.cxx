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

inline double eval_spline_float(const MySpline_float& s, const float x)
{
  int which = std::floor(x + 3);
  const float xknot = -3 + which;
  const float dx = x - xknot;
  return (s.ys[which]+dx*(s.bs[which]+dx*(s.cs[which]+dx*s.ds[which]))); 
}

inline double eval_splineAoS_float(const MySplineAoS_float& s, const float x)
{
  int which = std::floor(x + 3);
  const float xknot = -3 + which;
  const float dx = x - xknot;
  const Coeffs_float& c=s.c[which];
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

inline void eval_splineAVX_float(const MySplineAVX_float& s, float x, float* weights)
{
  int which = std::floor(x + 3);
  const float xknot = xs[which]; // -3 + which;
  // Set all of the items in the dx register to the same value
  __m256 dx = _mm256_set1_ps(x - xknot);

  // Choose the coefficients for the relevant knot (the same for all splines)
  __m256 y = s.ys[which];
  __m256 b = s.bs[which];
  __m256 c = s.cs[which];
  __m256 d = s.ds[which];

  // Calculate the weight (y + dx*(b + dx*(c + dx*d)))
  //
  // _m256_add_ps adds the two registers elementwise
  // _m256_mul_ps multiplies the two registers elementwise
  __m256 weight = _mm256_add_ps(c, _mm256_mul_ps(dx, d));
  weight = _mm256_add_ps(b, _mm256_mul_ps(dx, weight));
  weight = _mm256_add_ps(y, _mm256_mul_ps(dx, weight));

  // Store the output to `weights` (the 'u' in 'storeu' is for
  // 'unaligned', ie the output pointer is not necessarily aligned to
  // a 256-bit boundary)
  _mm256_storeu_ps(weights, weight);
}

inline void eval_splineAVX_float_estrin(const MySplineAVX_float& s, float x, float* weights)
{
  int which = std::floor(x + 3);
  const float xknot = xs[which]; // -3 + which;
  // Set all of the items in the dx register to the same value
  __m256 dx = _mm256_set1_ps(x - xknot);

  // Choose the coefficients for the relevant knot (the same for all splines)
  __m256 y = s.ys[which];
  __m256 b = s.bs[which];
  __m256 c = s.cs[which];
  __m256 d = s.ds[which];

  // Calculate the weight (y + dx*(b + dx*(c + dx*d)))
  //
  // _m256_add_ps adds the two registers elementwise
  // _m256_mul_ps multiplies the two registers elementwise
  __m256 t0 = _mm256_fmadd_ps(b, dx, y);
  __m256 t1 = _mm256_fmadd_ps(d, dx, c);
  __m256 dx2 = _mm256_mul_ps(dx, dx);
  __m256 weight = _mm256_fmadd_ps(t1, dx2, t0);
  // Store the output to `weights` (the 'u' in 'storeu' is for
  // 'unaligned', ie the output pointer is not necessarily aligned to
  // a 256-bit boundary)
  _mm256_storeu_ps(weights, weight);
}

inline void eval_splineAVX_float_fma(const MySplineAVX_float& s, float x, float* weights)
{
  int which = std::floor(x + 3);
  const double xknot = -3 + which;
  // Set all of the items in the dx register to the same value
  __m256 dx = _mm256_set1_ps(x - xknot);

  // Choose the coefficients for the relevant knot (the same for all splines)
  __m256 y = s.ys[which];
  __m256 b = s.bs[which];
  __m256 c = s.cs[which];
  __m256 d = s.ds[which];

  // Calculate the weight (y + dx*(b + dx*(c + dx*d)))
  //
  // _m256_add_ps adds the two registers elementwise
  // _m256_mul_ps multiplies the two registers elementwise
  __m256 weight = _mm256_fmadd_ps(dx, d, c);
  weight = _mm256_fmadd_ps(dx, weight, b);
  weight = _mm256_fmadd_ps(dx, weight, y);

  // Store the output to `weights` (the 'u' in 'storeu' is for
  // 'unaligned', ie the output pointer is not necessarily aligned to
  // a 256-bit boundary)
  _mm256_storeu_ps(weights, weight);
}

inline void eval_splineAVX512_float(const MySplineAVX512_float& s, float x, float* weights)
{
  int which = std::floor(x + 3);
  const float xknot = -3 + which;
  // Set all of the items in the dx register to the same value
  __m512 dx = _mm512_set1_ps(x - xknot);

  // Choose the coefficients for the relevant knot (the same for all splines)
  __m512 y = s.ys[which];
  __m512 b = s.bs[which];
  __m512 c = s.cs[which];
  __m512 d = s.ds[which];

  // Calculate the weight (y + dx*(b + dx*(c + dx*d)))
  //
  // _m512_add_ps adds the two registers elementwise
  // _m512_mul_ps multiplies the two registers elementwise
  //
  // the _ps suffix is for "packed single precision", ie the register
  // contains floats
  __m512 weight = _mm512_add_ps(c, _mm512_mul_ps(dx, d));
  weight = _mm512_add_ps(b, _mm512_mul_ps(dx, weight));
  weight = _mm512_add_ps(y, _mm512_mul_ps(dx, weight));

  // Store the output to `weights` (the 'u' in 'storeu' is for
  // 'unaligned', ie the output pointer is not necessarily aligned to
  // a 512-bit boundary)
  _mm512_storeu_ps(weights, weight);
}

MySpline_float make_spline_float(const MySpline &source)
{
  MySpline_float ret;
  for (int i=0; i<N; ++i) {
    ret.ys[i]=source.ys[i];
    ret.bs[i]=source.bs[i];
    ret.cs[i]=source.cs[i];
    ret.ds[i]=source.ds[i];
  }
  return ret;
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

MySplineAoS_float make_splineAoS_float(const MySpline& source)
{
  MySplineAoS_float s;
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

MySplineAVX_float make_splineAVX_float(const MySpline* source)
{
  MySplineAVX_float s;
  for (int i=0; i<N; ++i) {
    s.ys[i] = _mm256_setr_ps(source[0].ys[i], source[1].ys[i], source[2].ys[i], source[3].ys[i],
                             source[4].ys[i], source[5].ys[i], source[6].ys[i], source[7].ys[i]);
    s.bs[i] = _mm256_setr_ps(source[0].bs[i], source[1].bs[i], source[2].bs[i], source[3].bs[i],
                             source[4].bs[i], source[5].bs[i], source[6].bs[i], source[7].bs[i]);
    s.cs[i] = _mm256_setr_ps(source[0].cs[i], source[1].cs[i], source[2].cs[i], source[3].cs[i],
                             source[4].cs[i], source[5].cs[i], source[6].cs[i], source[7].cs[i]);
    s.ds[i] = _mm256_setr_ps(source[0].ds[i], source[1].ds[i], source[2].ds[i], source[3].ds[i],
                             source[4].ds[i], source[5].ds[i], source[6].ds[i], source[7].ds[i]);

  }

  return s;
}

MySplineAVX512_float make_splineAVX512_float(const MySpline* source)
{
  MySplineAVX512_float s;
  for (int i=0; i<N; ++i) {
    s.ys[i] = _mm512_setr_ps(source[0].ys[i],  source[1].ys[i],  source[2].ys[i],  source[3].ys[i],
                             source[4].ys[i],  source[5].ys[i],  source[6].ys[i],  source[7].ys[i],
                             source[8].ys[i],  source[9].ys[i],  source[10].ys[i], source[11].ys[i],
                             source[12].ys[i], source[13].ys[i], source[14].ys[i], source[15].ys[i]);
    s.bs[i] = _mm512_setr_ps(source[0].bs[i],  source[1].bs[i],  source[2].bs[i],  source[3].bs[i],
                             source[4].bs[i],  source[5].bs[i],  source[6].bs[i],  source[7].bs[i],
                             source[8].bs[i],  source[9].bs[i],  source[10].bs[i], source[11].bs[i],
                             source[12].bs[i], source[13].bs[i], source[14].bs[i], source[15].bs[i]);
    s.cs[i] = _mm512_setr_ps(source[0].cs[i],  source[1].cs[i],  source[2].cs[i],  source[3].cs[i],
                             source[4].cs[i],  source[5].cs[i],  source[6].cs[i],  source[7].cs[i],
                             source[8].cs[i],  source[9].cs[i],  source[10].cs[i], source[11].cs[i],
                             source[12].cs[i], source[13].cs[i], source[14].cs[i], source[15].cs[i]);
    s.ds[i] = _mm512_setr_ps(source[0].ds[i],  source[1].ds[i],  source[2].ds[i],  source[3].ds[i],
                             source[4].ds[i],  source[5].ds[i],  source[6].ds[i],  source[7].ds[i],
                             source[8].ds[i],  source[9].ds[i],  source[10].ds[i], source[11].ds[i],
                             source[12].ds[i], source[13].ds[i], source[14].ds[i], source[15].ds[i]);
  }

  return s;
}

// The current time in microseconds
uint64_t now_us()
{
  using namespace std::chrono;
  return duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
}

template<class T>
bool check_weights(const std::vector<T> &weights,
                   const std::vector<double> &weights_gold)

{
  int nbad=0;
  for (size_t i=0; i<weights.size(); ++i) {
    double gold = weights_gold[i];
    if (fabs(gold - weights[i]) > 1e-5) {
      ++nbad;
    }
  }
  if (nbad) {
    std::cout << "Disagreed at " << nbad << " places" << std::endl;
  }
  return nbad==0;
}

template<class T>
void benchmark(std::function<void(void)> fn,
               const std::vector<T> &weights,
               const std::vector<double> &weights_gold)
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
  std::cout << "good? " << check_weights(weights, weights_gold) << std::endl;
}


int main()
{
  const int nsplines = 8*1000*1000;

  // Spline objects for the 3 different methods
  std::vector<MySpline> splines(nsplines);
  std::vector<MySpline_float> splines_float(nsplines);
  std::vector<MySplineAoS_float> splinesAoS_float(nsplines);
  std::vector<MySplineAVX> splinesAVX(nsplines);
  std::vector<MySplineAVX_float> splinesAVX_float(nsplines);
  std::vector<MySplineAVX512_float> splinesAVX512_float(nsplines);

  // The systematic shifts
  std::vector<double> shifts(nsplines);
  std::vector<float> shifts_float(nsplines);

  // Weights from the three methods, and the "true" values from TSpline3
  std::vector<double> weights(nsplines);
  std::vector<double> weights_float(nsplines);
  std::vector<double> weightsAoS(nsplines);
  std::vector<double> weightsAVX(nsplines);
  std::vector<float>  weightsAVX_float(nsplines);
  std::vector<float>  weightsAVX_float_fma(nsplines);
  std::vector<float>  weightsAVX512_float(nsplines);
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
    splines_float[i] = make_spline_float(splines[i]);
    splinesAoS_float[i] = make_splineAoS_float(splines[i]);
        
    shifts[i] = *(reinterpret_cast<double*>(shift_buffer)+i);
    shifts_float[i] = shifts[i];
    weights_gold[i] = *(reinterpret_cast<double*>(weight_buffer)+i);
  }

  for(int i=0; i<nsplines/4; ++i) {
    splinesAVX[i] = make_splineAVX(&splines[i*4]);
  }
  for(int i=0; i<nsplines/8; ++i) {
    splinesAVX_float[i] = make_splineAVX_float(&splines[i*8]);
  }
  for(int i=0; i<nsplines/16; ++i) {
    splinesAVX512_float[i] = make_splineAVX512_float(&splines[i*16]);
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
  },
    weights, weights_gold);

  // -------------------------------------------------------------------
  std::cout << "Evaluating splines with MySpline_float" << std::endl;

  benchmark([&](){
    for (int i=0; i<nsplines; ++i) {
      weights_float[i] = eval_spline_float(splines_float[i], shifts_float[i]);
    }
  },
    weights_float, weights_gold);

  // -------------------------------------------------------------------
  std::cout << "Evaluating splines with MySplineAoS_float" << std::endl;
  benchmark([&](){
    for (int i=0; i<nsplines; ++i) {
      weightsAoS[i] = eval_splineAoS_float(splinesAoS_float[i], shifts_float[i]);
    }
  },
    weightsAoS, weights_gold);

  // -------------------------------------------------------------------
  std::cout << "Evaluating splines with eval_splineAVX" << std::endl;
  benchmark([&](){
    for (int i=0; i<nsplines/4; i++) {
      eval_splineAVX(splinesAVX[i], shifts[i*4], &weightsAVX[i*4]);
    }
  },
    weightsAVX, weights_gold);

  // -------------------------------------------------------------------
  std::cout << "Evaluating splines with eval_splineAVX_float" << std::endl;
  benchmark([&](){
    for (int i=0; i<nsplines/8; i++) {
      eval_splineAVX_float(splinesAVX_float[i], shifts_float[i*8], &weightsAVX_float[i*8]);
    }
  },
    weightsAVX_float, weights_gold);

  // -------------------------------------------------------------------
  std::cout << "Evaluating splines with eval_splineAVX_float_estrin" << std::endl;
  benchmark([&](){
    for (int i=0; i<nsplines/8; i++) {
      eval_splineAVX_float_estrin(splinesAVX_float[i], shifts_float[i*8], &weightsAVX_float[i*8]);
    }
  },
    weightsAVX_float, weights_gold);

  // -------------------------------------------------------------------
  std::cout << "Evaluating splines with eval_splineAVX_float unroll 4" << std::endl;
  benchmark([&](){
    for (int i=0; i<nsplines/8; i+=4) {
      eval_splineAVX_float(splinesAVX_float[i], shifts_float[i*8], &weightsAVX_float[i*8]);
      eval_splineAVX_float(splinesAVX_float[i+1], shifts_float[(i+1)*8], &weightsAVX_float[(i+1)*8]);
      eval_splineAVX_float(splinesAVX_float[i+2], shifts_float[(i+2)*8], &weightsAVX_float[(i+2)*8]);
      eval_splineAVX_float(splinesAVX_float[i+3], shifts_float[(i+3)*8], &weightsAVX_float[(i+3)*8]);
    }
  },
    weightsAVX_float, weights_gold);

  // -------------------------------------------------------------------
  std::cout << "Evaluating splines with eval_splineAVX_float_fma" << std::endl;
  benchmark([&](){
    for (int i=0; i<nsplines/8; i++) {
      eval_splineAVX_float_fma(splinesAVX_float[i], shifts_float[i*8], &weightsAVX_float_fma[i*8]);
    }
  },
    weightsAVX_float_fma, weights_gold);

  // -------------------------------------------------------------------
  std::cout << "Evaluating splines with eval_splineAVX512_float" << std::endl;
  benchmark([&](){
    for (int i=0; i<nsplines/16; i++) {
      eval_splineAVX512_float(splinesAVX512_float[i], shifts_float[i*16], &weightsAVX512_float[i*16]);
    }
  },
    weightsAVX512_float, weights_gold);

  // ===================================================================
  // std::cout << "Checking against TSpline3..." << std::flush;

  // bool good=check_weights(weights, weights_gold);
  // bool good_float=check_weights(weights_float, weights_gold);
  // bool goodAoS=check_weights(weightsAoS, weights_gold);
  // bool goodAVX=check_weights(weightsAVX, weights_gold);

  // std::cout << "MySpline: " << (good ? "good" : "bad") << "MySpline_float: " << (good_float ? "good" : "bad") << ", MySplineAoS: " << (goodAoS ? "good" : "bad")
  //           << ", AVX: " << (goodAVX ? "good" : "bad") << std::endl;
  
}
