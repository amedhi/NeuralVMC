/*---------------------------------------------------------------------------
* Author: Amal Medhi
* Date:   2017-01-16 22:12:57
* Last Modified by:   Amal Medhi, amedhi@macbook
* Last Modified time: 2017-03-27 23:49:33
* Copyright (C) Amal Medhi, amedhi@iisertvm.ac.in
*----------------------------------------------------------------------------*/
#ifndef DTYPE_H
#define DTYPE_H

#include <random>
#include <Eigen/Core>

namespace ann {
using random_engine = std::mt19937_64;
constexpr double pi(void) { return 3.1415926535897932384626433832795028841971693993751058209; }
}

#endif
