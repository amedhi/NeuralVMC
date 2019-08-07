/*---------------------------------------------------------------------------
* Author: Amal Medhi
* Date:   2017-01-30 14:51:12
* Last Modified by:   Amal Medhi, amedhi@macbook
* Last Modified time: 2017-04-11 10:28:23
* Copyright (C) Amal Medhi, amedhi@iisertvm.ac.in
*----------------------------------------------------------------------------*/
#ifndef WAVEFUNCTION_H
#define WAVEFUNCTION_H

#include <iostream>
#include <sstream>
#include <string>
#include <complex>
#include <vector>
//#include <map>
#include <memory>
#include <stdexcept>
#include <Eigen/Eigenvalues>
#include "../scheduler/task.h"
#include "./mf_model.h"
#include "./matrix.h"
#include "./identity.h"
#include "./fermisea.h"
#include "./bcs_state.h"

namespace var {

enum class wf_type {IDENTITY, FERMISEA, BCS};

// Wavefunction in 'Wannier space' (amplitudes in Wannier representation)
class Wavefunction 
{
public:
  //Wavefunction() {}
  Wavefunction(const lattice::LatticeGraph& graph, const input::Parameters& inputs,
    const bool& site_disorder=false);
  ~Wavefunction() {}
  const wf_type& type(void) const { return type_; }
  const std::string& name(void) const { return name_; }
  const VariationalParms& varparms(void) const { return groundstate_->varparms(); }
  std::string info_str(void) const { return groundstate_->info_str(); } 
  int compute(const lattice::LatticeGraph& graph, const input::Parameters& inputs, 
    const bool& psi_gradient=false);
  int compute(const lattice::LatticeGraph& graph, const var::parm_vector& pvector,
    const int& start_pos, const bool& psi_gradient=false);
  //int compute_gradients(const lattice::LatticeGraph& graph);
  const int& num_upspins(void) const { return groundstate_->num_upspins(); }
  const int& num_dnspins(void) const { return groundstate_->num_dnspins(); }
  const double& hole_doping(void) const { return groundstate_->hole_doping(); }
  std::string signature_str(void) const; 
  void get_vparm_names(std::vector<std::string>& names, int start_pos) const; 
  void get_vparm_values(var::parm_vector& values, int start_pos);
  void get_vparm_vector(std::vector<double>& vparm_values, int start_pos);
  void get_vparm_lbound(var::parm_vector& lbounds, int start_pos) const; 
  void get_vparm_ubound(var::parm_vector& ubounds, int start_pos) const; 
  void get_amplitudes(Matrix& psi, const std::vector<int>& row,  
    const std::vector<int>& col) const;
  void get_amplitudes(ColVector& psi_vec, const int& irow,  
    const std::vector<int>& col) const;
  void get_amplitudes(RowVector& psi_vec, const std::vector<int>& row,
    const int& icol) const;
  void get_amplitudes(amplitude_t& elem, const int& irow, const int& jcol) const;
  void get_gradients(Matrix& psi_grad, const int& n, 
    const std::vector<int>& row, const std::vector<int>& col) const;
private:
  std::unique_ptr<GroundState> groundstate_;
  wf_type type_;
  std::string name_;
  //bool pairing_type_{false};
  int num_sites_;
  // BCS_state bcs_state_;
  // FS_state fermisea_;
  Matrix psi_up_;
  Matrix psi_dn_;
  std::vector<Matrix> psi_gradient_;
  bool have_gradient_{false};
  // matrices & solvers
};


} // end namespace var

#endif