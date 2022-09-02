# multispline

This repo contains benchmarks for various methods of cubic spline evaluation. The motivating example is neutrino oscillation analysis with systematics implemented as event-by-event weights.

Requires ROOT 6.22 and cmake 3.23 as written (although you could probably change those without a problem by editing `CMakeLists.txt`). On a machine with cvmfs mounted, sourcing `setup.sh` will set up the necessary dependencies.

There are two executables: `make_random_inputs` will make a set of random splines and random systematic shifts (ie, points at which to evaluate the splines) and write them to file, while `multispline` will read the outputs from `make_random_inputs` and evaluate the splines at the given points, using various methods. The process is repeated tens of times for each method with the average and best times reported. The result of the spline evaluation is checked against a set of "golden" weights from ROOT's TSpline3 class.

The spline-evaluation methods include an AVX2-based method, which should work on nearly any modern x86 machine, and an AVX512-based method, which will only work on higher-end server class processors.
