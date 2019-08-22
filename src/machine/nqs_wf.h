/*---------------------------------------------------------------------------
* Copyright (C) 2019 by Amal Medhi <amedhi@iisertvm.ac.in>.
* @Author: Amal Medhi, amedhi@mbpro
* @Date:   2019-08-13 12:00:53
* @Last Modified by:   Amal Medhi, amedhi@mbpro
* @Last Modified time: 2019-08-13 12:01:25
*----------------------------------------------------------------------------*/
#ifndef NQS_WF_H
#define NQS_WF_H

#include "../scheduler/task.h"
#include <string>
#include "abstract_net.h"
#include "ffnet.h"

namespace nqs {

class NQS_Wavefunction 
{
public:
  NQS_Wavefunction() {}
  NQS_Wavefunction(const int& num_sites, const input::Parameters& inputs); 
  ~NQS_Wavefunction() {} 
  int init(const int& num_sites, const input::Parameters& inputs);
  const int& num_params(void) const;
  void init_parameters(ann::random_engine& rng, const double& sigma);
  void get_parm_names(std::vector<std::string>& pnames, const int& pos=0) const;
  void get_parm_values(ann::Vector& pvalues, const int& pos=0) const;
  void get_parm_vector(std::vector<double>& pvalues, const int& pos) const;
//  void update(const var::parm_vector& pvector, const unsigned& start_pos=0); 
  void update_parameters(const ann::Vector& pvalues, const int& pos);
  void update_state(const ann::ivector& fock_state);
  void update_state(const ann::ivector& fock_state, const std::vector<int> new_elems);
  const double& output(void) const; 
  double get_new_output(const ann::ivector& fock_state) const;
  void get_parm_lbound(eig::real_vec& lbound, const int& pos) const;
  void get_parm_ubound(eig::real_vec& ubound, const int& pos) const;
  void get_gradient(ann::Vector& grad, const int& pos) const;
private:
  std::unique_ptr<ann::AbstractNet> nnet_;
  std::string name_;
  int num_sites_{0};
};


} // end namespace nqs

#endif