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
//#define EIGEN_USE_MKL_ALL
#include <Eigen/Core>

//#define COMPLEX_PARAMETERS
namespace eig {
  using real_vec = Eigen::VectorXd;
  using cmpl_vec = Eigen::VectorXcd;
}

namespace ann {
using random_engine =	std::mt19937_64;
using dtype = double;
using ivector = Eigen::VectorXi;
using Vector = Eigen::VectorXd;
using RowVector = Eigen::RowVectorXd;
using Matrix = Eigen::MatrixXd;
using Array1D = Eigen::ArrayXd;
using Array2D = Eigen::Array<double,Eigen::Dynamic,Eigen::Dynamic>;

} // end namespace nqs

#endif
