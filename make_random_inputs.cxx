#include "multispline.h"

#include "TRandom3.h"
#include "TSpline.h"

#include <array>
#include <fstream>
#include <iostream>

std::array<double, N> random_ys()
{
  std::array<double, N> ret;
  for (int i=0; i<N; ++i) {
    ret[i] = (i-3)*gRandom->Uniform();
  }
  return ret;
}

MySpline make_spline(std::array<double, N> ys)
{
  TSpline3 spline("title", xs, ys.data(), N);
  MySpline s;
  for (int i=0; i<N; ++i) {
    double x, y, b, c, d;
    spline.GetCoeff(i, x, y, b, c, d);
    s.ys[i]=y;
    s.bs[i]=b;
    s.cs[i]=c;
    s.ds[i]=d;
  }

  return s;
}

int main()
{
  gRandom->SetSeed(42);
  
  const int nsplines = 10*1000*1000;
  std::vector<MySpline> splines(nsplines);
  
  std::cout << "Making random splines" << std::endl;
  
  for (int i=0; i<nsplines; ++i) {
    auto the_ys = random_ys();
    splines[i] = make_spline(the_ys);
  }

  std::ofstream fout("spline_coeffs.dat", std::ios::binary);
  fout.write(reinterpret_cast<const char*>(splines.data()), splines.size()*sizeof(MySpline));

  std::ofstream fout_shifts("shifts.dat", std::ios::binary);
  std::ofstream fout_weights("weights.dat", std::ios::binary);

  double shift=0;
  for (int i=0; i<nsplines; ++i) {
    TSpline3 spline3("title", xs, splines[i].ys, N);
    // 4 shifts in a row are the same, to simulate the fact that there
    // are multiple splines for a given systematic shift, eg for
    // different modes, or bins in a histogram. 4 is chosen because
    // it's the number of doubles in an AVX2 register
    if (i%4 == 0) {
      shift = gRandom->Uniform(5.99) - 2.99;
    }
    double weight = spline3.Eval(shift);

    fout_shifts.write(reinterpret_cast<const char*>(&shift), sizeof(shift));
    fout_weights.write(reinterpret_cast<const char*>(&weight), sizeof(weight));
  }
}
