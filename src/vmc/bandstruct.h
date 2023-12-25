/*---------------------------------------------------------------------------
* Author: Amal Medhi
* Date:   2023-08-25 21:41:40
* Last Modified by:   Amal Medhi, amedhi@macbook
* Last Modified time: 2023-08-25 21:41:40
* Copyright (C) Amal Medhi, amedhi@iisertvm.ac.in
*----------------------------------------------------------------------------*/
#ifndef OBS_BANDSTURCT_H
#define OBS_BANDSTURCT_H

#include "../mcdata/mc_observable.h"
#include "../lattice/lattice.h"
#include "./sysconfig.h"

namespace vmc {

class BandStruct : public mcdata::MC_Observable
{
public:
  using MC_Observable::MC_Observable;
  void reset(void) override;
  void setup(const lattice::Lattice& lattice, const SysConfig& config);
  void measure(const lattice::Lattice& lattice, const SysConfig& config);
  void print_heading(const std::string& header, 
    const std::vector<std::string>& xvars) override;
  void print_result(const std::vector<double>& xvals) override; 
private:
  bool setup_done_{false};
  bool computation_done_{false};
  int num_bands_{1};
  std::vector<Vector3d> symm_kpoints_;
  std::vector<int> symm_pidx_;
  std::vector<std::string> symm_pname_;
  std::vector<std::string> xvars_;
  Eigen::MatrixXd Ekn_;
};


} // end namespace vmc

#endif