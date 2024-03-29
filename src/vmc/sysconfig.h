/*---------------------------------------------------------------------------
* Author: Amal Medhi
* Date:   2017-02-18 13:54:54
* Last Modified by:   Amal Medhi, amedhi@macbook
* Last Modified time: 2017-05-10 23:35:53
* Copyright (C) Amal Medhi, amedhi@iisertvm.ac.in
*----------------------------------------------------------------------------*/
#ifndef SYSCONFIG_H
#define SYSCONFIG_H

#include "../scheduler/worker.h"
#include "../lattice/lattice.h"
#include "../lattice/graph.h"
#include "../model/model.h"
#include "../wavefunction/wavefunction.h"
#include "../wavefunction/projector.h"
#include "../matrix/matrix.h"
#include "../machine/nqs_wf.h"
#include "./basisstate.h"

namespace vmc {

constexpr double dratio_cutoff(void) { return 1.0E-8; } 
constexpr double gfactor_cutoff(void) { return 1.0E-8; } 

class SysConfig 
{
public:
  SysConfig(const input::Parameters& parms, const lattice::Lattice& lattice, 
    const model::Hamiltonian& model);
  ~SysConfig() {}
  int init_files(const std::string& prefix, const input::Parameters& inputs);
  std::string info_str(void) const; 
  int build(const lattice::Lattice& lattice, const input::Parameters& inputs, 
    const bool& with_gradient=false);
  int build(const lattice::Lattice& lattice, const var::parm_vector& vparms, 
    const bool& need_psi_grad=false);
  int rebuild(const lattice::Lattice& lattice);
  RandomGenerator& rng(void) const { return fock_basis_.rng(); }
  std::string signature_str(void) const { return wf_.signature_str(); } 
  const int& num_varparms(void) const { return num_varparms_; } 
  void get_varp_values(RealVector& varp_values) const; 
  void get_varp_lbound(RealVector& varp_lbound) const;
  void get_varp_ubound(RealVector& varp_ubound) const;
  void get_varp_names(std::vector<std::string>& varp_names) const;
  std::vector<std::string> varp_names(void) const;
  const double& hole_doping(void) const { return wf_.hole_doping(); }
  int save_parameters(const var::parm_vector& pvector); 
  int num_particles(void) const { return num_upspins_+num_dnspins_; }
  int update_state(void);
  double accept_ratio(void);
  void reset_accept_ratio(void);
  int apply_niup_nidn(const int& site_i) const;
  int apply_ni_dblon(const int& site_i) const;
  int apply_ni_holon(const int& site_i) const;
  amplitude_t apply_cdagc_up(const int& i_fr, const int& j_to,
    const int& bc_state, const std::complex<double>& bc_phase) const;
  amplitude_t apply_cdagc_dn(const int& i_fr, const int& j_to,
    const int& bc_state, const std::complex<double>& bc_phase) const;
  int apply(const model::op::quantum_op& qn_op, const int& site_i) const;
  amplitude_t apply(const model::op::quantum_op& op, const int& site_i, 
    const int& site_j, const int& bc_state, const std::complex<double>& bc_phase) const;
  amplitude_t apply_bondsinglet_hop(const int& fr_site_i, const int& fr_site_ia, 
    const int& to_site_j, const int& to_site_jb) const;
  amplitude_t apply_sitepair_hop(const int& fr_site, const int& to_site) const;
  void get_grad_logpsi(Vector& grad_logpsi) const;
  const int& num_updates(void) const { return num_updates_; }
  const var::Wavefunction& wavefunc(void) const { return wf_; }
  void print_stats(std::ostream& os=std::cout) const;
  //var::VariationalParms& var_parms(void) { return wf.var_parms(); }
private:
  FockBasis fock_basis_;
  var::WavefunProjector pj_;
  //var::FFN_State ffnet_;
  var::Wavefunction wf_;
  nqs::NQS_Wavefunction nqs_;
  int nqs_sign_{1};
  //double ffn_psi_;
  amplitude_t nqs_psi_;
  bool have_mf_part_{false};
  Matrix psi_mat_;
  Matrix psi_inv_;
  mutable ColVector psi_row_;
  mutable RowVector psi_col_;
  mutable RowVector inv_row_;
  mutable Matrix psi_grad_;
  int num_sites_;
  int num_upspins_;
  int num_dnspins_;

  // no of variational parameters
  int num_pj_parms_{0};
  int num_wf_parms_{0};
  int num_mf_parms_{0};
  int num_nn_parms_{0};
  int num_varparms_{0};

  // variational parameters prefix folder
  bool load_parms_from_file_{false};
  //std::string save_path_;
  //std::string load_path_;

  // mc parameters
  enum move_t {uphop, dnhop, exch, end};
  int num_updates_{0};
  int num_iterations_{0};
  //int num_total_steps_{0};
  int num_uphop_moves_{0};
  int num_dnhop_moves_{0};
  int num_exchange_moves_{0};
  int refresh_cycle_{100};
  long num_proposed_moves_[move_t::end];
  long num_accepted_moves_[move_t::end];
  long last_proposed_moves_;
  long last_accepted_moves_;

  // helper methods
  int init_config(void);
  int set_run_parameters(void);
  int do_upspin_hop(void);
  int do_dnspin_hop(void);
  int do_spin_exchange(void);
  int inv_update_upspin(const int& upspin, const ColVector& psi_row, 
    const amplitude_t& det_ratio);
  int inv_update_dnspin(const int& dnspin, const RowVector& psi_col, 
    const amplitude_t& det_ratio);
  amplitude_t apply_cdagc2_up(const int& i, const int& j,
    const int& bc_state, const std::complex<double>& bc_phase) const;
  amplitude_t apply_cdagc2_dn(const int& i, const int& j,
    const int& bc_state, const std::complex<double>& bc_phase) const;
  amplitude_t apply_sisj_plus(const int& i, const int& j) const;
};

} // end namespace vmc

#endif