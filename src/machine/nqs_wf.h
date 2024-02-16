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
#include "../matrix/matrix.h"
#include "../lattice/lattice.h"
#include "abstract_net.h"
#include "inet.h"
#include "ffnet.h"
#include "rbm.h"

namespace nqs {

enum class net_id {INET, FFNN, FFNN_SIGN, FFNN_SIGN2, RBM};

constexpr std::complex<double> ii(void) { return std::complex<double>(0.0, 1.0); }

class NQS_Wavefunction 
{
public:
  NQS_Wavefunction() {}
  NQS_Wavefunction(const lattice::Lattice& lattice, const input::Parameters& inputs); 
  ~NQS_Wavefunction() {} 
  int init(const lattice::Lattice& lattice, const input::Parameters& inputs);
  const int& num_params(void) const;
  bool is_exponential_type(void) const { return exponential_type_; }
  void init_parameter_file(const std::string& save_path, const std::string& load_path); 
  void init_parameters(ann::random_engine& rng, const double& sigma);
  void save_parameters(void) const;
  void load_parameters(void);
  void get_varp_names(std::vector<std::string>& pnames, const int& pos=0) const;
  void get_varp_values(RealVector& pvalues, const int& pos=0) const;
//  void update(const var::parm_vector& pvector, const unsigned& start_pos=0); 
  void update_parameters(const RealVector& pvalues, const int& pos);
  void update_parameter(const int& id, const double& value);
  void update_state(const IntVector& fock_state);
  void update_state(const IntVector& fock_state, const std::vector<int> new_elems);
  const amplitude_t& output(void) const; 
  amplitude_t get_new_output(const IntVector& fock_state) const;
  amplitude_t get_new_output(const IntVector& fock_state, const std::vector<int> new_elems) const;
  amplitude_t get_new_output_ratio(const IntVector& fock_state) const;
  amplitude_t get_new_output_ratio(const IntVector& fock_state, const std::vector<int> new_elems) const;
  void get_parm_lbound(RealVector& lbound, const int& pos) const;
  void get_parm_ubound(RealVector& ubound, const int& pos) const;
  void get_gradient(Vector& grad, const int& pos) const;
  void get_log_gradient(Vector& grad, const int& pos) const;
private:
  net_id nid_{net_id::FFNN};
  std::unique_ptr<ann::AbstractNet> nnet_;
  std::unique_ptr<ann::AbstractNet> sign_nnet_;
  std::string name_;
  bool exponential_type_{false};
  bool complex_type_{false};
  int num_sites_{0};
  int num_params_{0};
  int num_output_units_{0};
  bool have_sign_nnet_{false};
  mutable amplitude_t output_{0.0};
  mutable RealMatrix gradient_mat_;
};


} // end namespace nqs

#endif