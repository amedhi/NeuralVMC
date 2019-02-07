/*---------------------------------------------------------------------------
* Author: Amal Medhi
* Date:   2017-02-18 14:01:12
* Last Modified by:   Amal Medhi, amedhi@macbook
* Last Modified time: 2017-05-20 11:16:31
* Copyright (C) Amal Medhi, amedhi@iisertvm.ac.in
*----------------------------------------------------------------------------*/
#include "./sysconfig.h"
#include <Eigen/SVD>

namespace vmc {

SysConfig::SysConfig(const input::Parameters& inputs, 
  const lattice::LatticeGraph& graph, const model::Hamiltonian& model)
  : BasisState(graph.num_sites(), model.double_occupancy())
  , fock_basis_(graph.num_sites(), model.double_occupancy())
  , pj(inputs)
  , ffnet_(graph.num_sites(), inputs)
  , wf(graph, inputs)
  , num_sites_(graph.num_sites())
{
  // variational parameters
  num_pj_parms_ = pj.varparms().size();
  num_wf_parms_ = wf.varparms().size();
  num_varparms_ = (num_pj_parms_ + num_wf_parms_);
  vparm_names_.resize(num_varparms_);
  vparm_lbound_.resize(num_varparms_);
  vparm_ubound_.resize(num_varparms_);
  // names
  pj.get_vparm_names(vparm_names_,0);
  wf.get_vparm_names(vparm_names_,num_pj_parms_);
  // values are not static and may change
  // bounds
  pj.get_vparm_lbound(vparm_lbound_,0);
  wf.get_vparm_lbound(vparm_lbound_,num_pj_parms_);
  pj.get_vparm_ubound(vparm_ubound_,0);
  wf.get_vparm_ubound(vparm_ubound_,num_pj_parms_);
}

const var::parm_vector& SysConfig::vparm_values(void) 
{
  // values as 'var::parm_vector'
  vparm_values_.resize(num_varparms_);
  pj.get_vparm_values(vparm_values_,0);
  wf.get_vparm_values(vparm_values_,num_pj_parms_);
  return vparm_values_;
}

const std::vector<double>& SysConfig::vparm_vector(void) 
{
  // values as 'std::double'
  vparm_vector_.resize(num_varparms_);
  pj.get_vparm_vector(vparm_vector_,0);
  wf.get_vparm_vector(vparm_vector_,num_pj_parms_);
  return vparm_vector_;
}

int SysConfig::build(const lattice::LatticeGraph& graph, const input::Parameters& inputs,
    const bool& with_gradient)
{
  if (num_sites_==0) return -1;
  pj.update(inputs);
  wf.compute(graph, inputs, with_gradient);
  init_config();
  return 0;
}

int SysConfig::build(const lattice::LatticeGraph& graph, const var::parm_vector& pvector,
  const bool& need_psi_grad)
{
  if (num_sites_==0) return -1;
  pj.update(pvector, 0);
  unsigned start_pos = pj.varparms().size();
  wf.compute(graph, pvector, start_pos, need_psi_grad);
  init_config();
  return 0;
}

int SysConfig::init_config(void)
{
  num_upspins_ = wf.num_upspins();
  num_dnspins_ = wf.num_dnspins();
  if (num_upspins_==0 && num_dnspins_==0) return -1;
  if (num_upspins_ != num_dnspins_) 
    throw std::range_error("*SysConfig::init_config: unequal UP & DN spin case not implemented");
  fock_basis_.init_spins(num_upspins_,num_dnspins_);
  // run parameters
  ffnet_.update_state(fock_basis_);
  ffn_psi_ = ffnet_.output();
  set_run_parameters();
  return 0;
}

int SysConfig::set_run_parameters(void)
{
  num_updates_ = 0;
  refresh_cycle_ = 100;
  // number of moves per mcstep
  int n_up = static_cast<int>(num_upspins_);
  int n_dn = static_cast<int>(num_dnspins_);
  if (double_occupancy()) {
    num_uphop_moves_ = num_upspins_;
    num_dnhop_moves_ = num_dnspins_;
    num_exchange_moves_ = std::min(n_up, n_dn);
    //num_exchange_moves_ = 2*std::min(n_up, n_dn);
  }
  else {
    int num_holes = num_sites_-(num_upspins_+num_dnspins_);
    num_uphop_moves_ = std::min(n_up, num_holes);
    num_dnhop_moves_ = std::min(n_dn, num_holes);
    num_exchange_moves_ = std::min(n_up, n_dn);
    //num_exchange_moves_ = 4*std::min(n_up, n_dn);
  }
  for (int i=0; i<move_t::end; ++i) {
    num_proposed_moves_[i] = 0;
    num_accepted_moves_[i] = 0;
  }
  last_proposed_moves_ = 1;
  last_accepted_moves_ = 1;

  // work arrays 
  psi_row.resize(num_dnspins_);
  psi_col.resize(num_upspins_);
  inv_row.resize(num_upspins_);
  psi_grad.resize(num_upspins_,num_dnspins_);
  return 0;
}

int SysConfig::update_state(void)
{
  for (int n=0; n<num_uphop_moves_; ++n) do_upspin_hop();
  for (int n=0; n<num_dnhop_moves_; ++n) do_dnspin_hop();
  for (int n=0; n<num_exchange_moves_; ++n) do_spin_exchange();
  num_updates_++;
  if (num_updates_ % refresh_cycle_ == 0) {
    psi_inv = psi_mat.inverse();
  }
  return 0;
}

int SysConfig::do_upspin_hop(void)
{
  std::cout << "\n state=" << fock_basis_.transpose() << "\n";
  if (fock_basis_.gen_upspin_hop()) {
    num_proposed_moves_[move_t::uphop]++;
    last_proposed_moves_++;
  std::cout << "\n state=" << fock_basis_.transpose() << "\n";
    //wf.get_amplitude(fock_basis_.trial_state());
    //wf.get_amplitude(fock_basis_);
    double new_psi = ffnet_.get_output(fock_basis_);
    std::cout << "\n psi-1=" << ffn_psi_ << "\n";
    std::cout << "psi-2=" << new_psi << "\n";
    amplitude_t det_ratio = new_psi/ffn_psi_;
    double proj_ratio = 1.0;
    amplitude_t weight_ratio = det_ratio * proj_ratio;
    double transition_proby = std::norm(weight_ratio);
    if (rng().random_real()<transition_proby) {
      num_accepted_moves_[move_t::uphop]++;
      last_accepted_moves_++;
      // upddate state
      fock_basis_.commit_last_move();
    }
    else {
      fock_basis_.undo_last_move();
    }
  } 
  throw std::range_error("Exiting at: SysConfig::do_upspin_hop");

  return 0;
}

int SysConfig::do_dnspin_hop(void)
{
  int dnspin, to_site;
  std::tie(dnspin, to_site) = gen_dnspin_hop();
  if (to_site < 0) return 0; // valid move not found
  num_proposed_moves_[move_t::dnhop]++;
  last_proposed_moves_++;
  // new col for this move
  wf.get_amplitudes(psi_col, upspin_sites(), to_site);
  amplitude_t det_ratio = psi_col.cwiseProduct(psi_inv.row(dnspin)).sum();
  if (std::abs(det_ratio) < dratio_cutoff()) return 0; // for safety
  double proj_ratio = pj.gw_ratio(dblocc_increament());
  amplitude_t weight_ratio = det_ratio * proj_ratio;
  double transition_proby = std::norm(weight_ratio);
  if (rng().random_real()<transition_proby) {
    num_accepted_moves_[move_t::dnhop]++;
    last_accepted_moves_++;
    // upddate state
    accept_last_move();
    // update amplitudes
    inv_update_dnspin(dnspin,psi_col,det_ratio);
  }
  return 0;
}

int SysConfig::do_spin_exchange(void)
{
  int upspin, up_tosite;
  std::tie(upspin, up_tosite) = gen_exchange_move();
  if (up_tosite < 0) return 0; // valid move not found
  num_proposed_moves_[move_t::exch]++;
  last_proposed_moves_++;
  // for upspin hop forward
  wf.get_amplitudes(psi_row, up_tosite, dnspin_sites());
  amplitude_t det_ratio1 = psi_row.cwiseProduct(psi_inv.col(upspin)).sum();
  if (std::abs(det_ratio1) < dratio_cutoff()) return 0; // for safety

  // now for dnspin hop backward
  int dnspin, dn_tosite;
  std::tie(dnspin, dn_tosite) = exchange_move_part();
  // new col for this move
  wf.get_amplitudes(psi_col, upspin_sites(), dn_tosite);
  // since the upspin should have moved
  wf.get_amplitudes(psi_col(upspin), up_tosite, dn_tosite);
  // updated 'dnspin'-th row of psi_inv
  amplitude_t ratio_inv = amplitude_t(1.0)/det_ratio1;
  // elements other than 'upspin'-th
  for (int i=0; i<upspin; ++i) {
    amplitude_t beta = ratio_inv*psi_row.cwiseProduct(psi_inv.col(i)).sum();
    inv_row(i) = psi_inv(dnspin,i) - beta * psi_inv(dnspin,upspin);
  }
  for (int i=upspin+1; i<static_cast<int>(num_upspins_); ++i) {
    amplitude_t beta = ratio_inv*psi_row.cwiseProduct(psi_inv.col(i)).sum();
    inv_row(i) = psi_inv(dnspin,i) - beta * psi_inv(dnspin,upspin);
  }
  inv_row(upspin) = ratio_inv * psi_inv(dnspin,upspin);
  // ratio for the dnspin hop
  amplitude_t det_ratio2 = psi_col.cwiseProduct(inv_row).sum();
  if (std::abs(det_ratio2) < dratio_cutoff()) return 0; // for safety

  // weight ratio (gw pj does not play here)
  amplitude_t weight_ratio = det_ratio1 * det_ratio2;
  double transition_proby = std::norm(weight_ratio);
  //std::cout << "W = " << transition_proby << "\n";
  if (rng().random_real()<transition_proby) {
    num_accepted_moves_[move_t::exch]++;
    last_accepted_moves_++;
    // upddate state
    accept_last_move();
    // update amplitudes
    inv_update_upspin(upspin,psi_row,det_ratio1);
    inv_update_dnspin(dnspin,psi_col,det_ratio2);
  }
  return 0;
}

int SysConfig::inv_update_upspin(const unsigned& upspin, const ColVector& psi_row, 
  const amplitude_t& det_ratio)
{
  psi_mat.row(upspin) = psi_row;
  amplitude_t ratio_inv = amplitude_t(1.0)/det_ratio;
  for (unsigned i=0; i<upspin; ++i) {
    amplitude_t beta = ratio_inv*psi_row.cwiseProduct(psi_inv.col(i)).sum();
    psi_inv.col(i) -= beta * psi_inv.col(upspin);
  }
  for (unsigned i=upspin+1; i<num_upspins_; ++i) {
    amplitude_t beta = ratio_inv*psi_row.cwiseProduct(psi_inv.col(i)).sum();
    psi_inv.col(i) -= beta * psi_inv.col(upspin);
  }
  psi_inv.col(upspin) *= ratio_inv;
  return 0;
}

int SysConfig::inv_update_dnspin(const unsigned& dnspin, const RowVector& psi_col, 
  const amplitude_t& det_ratio)
{
  psi_mat.col(dnspin) = psi_col;
  amplitude_t ratio_inv = amplitude_t(1.0)/det_ratio;
  for (unsigned i=0; i<dnspin; ++i) {
    amplitude_t beta = ratio_inv*psi_col.cwiseProduct(psi_inv.row(i)).sum();
    psi_inv.row(i) -= beta * psi_inv.row(dnspin);
  }
  for (unsigned i=dnspin+1; i<num_dnspins_; ++i) {
    amplitude_t beta = ratio_inv*psi_col.cwiseProduct(psi_inv.row(i)).sum();
    psi_inv.row(i) -= beta * psi_inv.row(dnspin);
  }
  psi_inv.row(dnspin) *= ratio_inv;
  return 0;
}


amplitude_t SysConfig::apply(const model::op::quantum_op& qn_op, const unsigned& site_i, 
    const unsigned& site_j, const int& bc_phase) const
{
  amplitude_t term(0); 
  switch (qn_op.id()) {
    case model::op_id::cdagc_sigma:
      term = apply_upspin_hop(site_i,site_j,bc_phase);
      term+= apply_dnspin_hop(site_i,site_j,bc_phase);
      break;
    case model::op_id::sisj_plus:
      term = apply_sisj_plus(site_i,site_j); 
      break;
    default: 
      throw std::range_error("SysConfig::apply: undefined bond operator.");
  }
  return term;
}

int SysConfig::apply(const model::op::quantum_op& qn_op, const unsigned& site_i) const
{
  switch (qn_op.id()) {
    case model::op_id::ni_sigma:
      return operator[](site_i).count();
    //case model::op_id::ni_up:
    //  return static_cast<int>(operator[](site_i).have_upspin());
    //case model::op_id::ni_dn:
    //  return static_cast<int>(operator[](site_i).have_dnspin());
    case model::op_id::niup_nidn:
      return apply_niup_nidn(site_i); 
    default: 
      throw std::range_error("SysConfig::apply: undefined site operator");
  }
}

int SysConfig::apply_niup_nidn(const unsigned& site_i) const
{
  if (operator[](site_i).count()==2) return 1;
  else return 0;
}

amplitude_t SysConfig::apply_upspin_hop(const unsigned& i, const unsigned& j,
  const int& bc_phase) const
{
  //if (i == j) return ampl_part(1.0);
  if (i == j) return ampl_part(operator[](i).num_upspins());
  int upspin, to_site;
  int delta_nd;
  //check_upspin_hop(i,j)
  const SiteState* state_i = &operator[](i);
  const SiteState* state_j = &operator[](j);
  if (state_i->have_upspin() && state_j->have_uphole()) {
    upspin = state_i->upspin_id();
    to_site = j;
    delta_nd = state_j->count(); // must be 0 or one
    if (state_i->count()==2) delta_nd--;
  }
  else if (state_j->have_upspin() && state_i->have_uphole()) {
    upspin = state_j->upspin_id();
    to_site = i;
    delta_nd = state_i->count(); // must be 0 or one
    if (state_j->count()==2) delta_nd--;
  }
  else {
    return amplitude_t(0.0);
  }
  // site occupancy constraint
  if (operator[](to_site).count()==site_capacity()) return amplitude_t(0.0);

  // det_ratio for the term
  wf.get_amplitudes(psi_row, to_site, dnspin_sites());
  amplitude_t det_ratio = psi_row.cwiseProduct(psi_inv.col(upspin)).sum();
  det_ratio = ampl_part(std::conj(det_ratio));
  return amplitude_t(bc_phase) * det_ratio * pj.gw_ratio(delta_nd);
}

amplitude_t SysConfig::apply_dnspin_hop(const unsigned& i, const unsigned& j,
  const int& bc_phase) const
{
  //if (i == j) return amplitude_t(1.0);
  if (i == j) return ampl_part(operator[](i).num_dnspins());
  int dnspin, to_site;
  int delta_nd;
  const SiteState* state_i = &operator[](i);
  const SiteState* state_j = &operator[](j);
  if (state_i->have_dnspin() && state_j->have_dnhole()) {
    dnspin = state_i->dnspin_id();
    to_site = j;
    delta_nd = state_j->count(); // must be 0 or one
    if (state_i->count()==2) delta_nd--;
  }
  else if (state_j->have_dnspin() && state_i->have_dnhole()) {
    dnspin = state_j->dnspin_id();
    to_site = i;
    delta_nd = state_i->count(); // must be 0 or one
    if (state_j->count()==2) delta_nd--;
  }
  else {
    return amplitude_t(0.0);
  }
  // site occupancy constraint
  if (operator[](to_site).count()==site_capacity()) return amplitude_t(0.0);

  // det_ratio for the term
  wf.get_amplitudes(psi_col, upspin_sites(), to_site);
  amplitude_t det_ratio = psi_col.cwiseProduct(psi_inv.row(dnspin)).sum();
  det_ratio = ampl_part(std::conj(det_ratio));
  return amplitude_t(bc_phase) * det_ratio * pj.gw_ratio(delta_nd);
}

amplitude_t SysConfig::apply_sisj_plus(const unsigned& i, const unsigned& j) const
{
/* It evaluates the following operator:
 !   O = (S_i.S_j - (n_i n_j)/4)
 ! The operator can be cast in the form,
 !   O = O_{ud} + O_{du}
 ! where,
 !   O_{ud} = 1/2*(- c^{\dag}_{j\up}c_{i\up} c^{\dag}_{i\dn}c_{j\dn}
 !                 - n_{i\up} n_{j_dn})
 ! O_{du} is obtained from O_{ud} by interchanging spin indices. */

  const SiteState* state_i = &operator[](i);
  const SiteState* state_j = &operator[](j);
  // ni_nj term
  double ninj_term;
  if (state_i->have_upspin() && state_j->have_dnspin()) 
    ninj_term = -0.5;
  else if (state_i->have_dnspin() && state_j->have_upspin()) 
    ninj_term = -0.5;
  else ninj_term = 0.0;

  // spin exchange term
  // if any of the two sites doubly occupied, no exchange possible
  if (state_i->count()==2 || state_j->count()==2) return amplitude_t(ninj_term);

  unsigned upspin, up_tosite;
  unsigned dnspin, dn_tosite;
  if (state_i->have_upspin() && state_j->have_dnspin()) {
    upspin = state_i->upspin_id();
    up_tosite = j;
    dnspin = state_j->dnspin_id();
    dn_tosite = i;
  }
  else if (state_i->have_dnspin() && state_j->have_upspin()) {
    upspin = state_j->upspin_id();
    up_tosite = i;
    dnspin = state_i->dnspin_id();
    dn_tosite = j;
  }
  else return amplitude_t(ninj_term);

  // det_ratio for the term
  wf.get_amplitudes(psi_row, up_tosite, dnspin_sites());
  amplitude_t det_ratio1 = psi_row.cwiseProduct(psi_inv.col(upspin)).sum();
  // now for dnspin hop 
  wf.get_amplitudes(psi_col, upspin_sites(), dn_tosite);
  // since the upspin should have moved
  wf.get_amplitudes(psi_col(upspin), up_tosite, dn_tosite);
  // updated 'dnspin'-th row of psi_inv
  amplitude_t ratio_inv = amplitude_t(1.0)/det_ratio1;

  // for safety: if 'det_ratio1 == 0', result is zero
  if (std::isinf(std::abs(ratio_inv))) {
    return amplitude_t(ninj_term);
  }

  // elements other than 'upspin'-th
  for (unsigned i=0; i<upspin; ++i) {
    amplitude_t beta = ratio_inv*psi_row.cwiseProduct(psi_inv.col(i)).sum();
    inv_row(i) = psi_inv(dnspin,i) - beta * psi_inv(dnspin,upspin);
  }
  for (unsigned i=upspin+1; i<num_upspins_; ++i) {
    amplitude_t beta = ratio_inv*psi_row.cwiseProduct(psi_inv.col(i)).sum();
    inv_row(i) = psi_inv(dnspin,i) - beta * psi_inv(dnspin,upspin);
  }
  inv_row(upspin) = ratio_inv * psi_inv(dnspin,upspin);
  // ratio for the dnspin hop
  amplitude_t det_ratio2 = psi_col.cwiseProduct(inv_row).sum();
  amplitude_t det_ratio = ampl_part(std::conj(det_ratio1*det_ratio2));
  /*
  if (std::isnan(det_ratio)) {
    std::cout << std::scientific<< det_ratio1 << "\n\n";
    std::cout << std::scientific<< ratio_inv << "\n\n";
    std::cout << std::scientific<< det_ratio2 << "\n\n";
    std::cout << "NaN detected\n"; getchar();
  }*/
  return -0.5 * det_ratio + amplitude_t(ninj_term);
}

amplitude_t SysConfig::apply_bondsinglet_hop(const unsigned& i_dag, 
  const unsigned& ia_dag, const int& bphase_i, const unsigned& j, 
  const unsigned& jb, const int& bphase_j) const
{
  // Evaluates the following operator:
  //   F_{ab}(i,j) = (c^{\dag}_{i\up}c^{\dag}_{i+a\dn} -  c^{\dag}_{i\dn}c^{\dag}_{i+a,\up})/sqrt(2)
  //          x (c_{j+b\dn}c_{j\up} - c_{j+b\up}c_{j\dn})/sqrt(2)
  //          = 0.5 * [c^{\dag}_{i\up}c_{j\up} x c^{\dag}_{i+a\dn}c_{j+b\dn}
  //                 + c^{\dag}_{i+a\up}c_{j\up} x c^{\dag}_{i\dn}c_{j+b\dn}
  //                 + c^{\dag}_{i+a\up}c_{j+b\up} x c^{\dag}_{i\dn}c_{j\dn}
  //                 + c^{\dag}_{i\up}c_{j+b\up} x c^{\dag}_{i+a\dn}c_{j\dn}]

  int num_terms = 4;
  unsigned up_fromsite, up_tosite;
  unsigned dn_fromsite, dn_tosite;
  amplitude_t net_ratio(0.0);
  for (int iterm=0; iterm<num_terms; ++iterm) {
    switch (iterm) {
      case 0:
        up_fromsite = j; up_tosite = i_dag;
        dn_fromsite = jb; dn_tosite = ia_dag;
        break;
      case 1:
        up_fromsite = j; up_tosite = ia_dag;
        dn_fromsite = jb; dn_tosite = i_dag;
        break;
      case 2:
        up_fromsite = jb; up_tosite = ia_dag;
        dn_fromsite = j; dn_tosite = i_dag;
        break;
      case 3:
        up_fromsite = jb; up_tosite = i_dag;
        dn_fromsite = j; dn_tosite = ia_dag;
        break;
    }
    const SiteState* state_j = &operator[](up_fromsite);
    const SiteState* state_i = &operator[](up_tosite);
    const SiteState* state_jb = &operator[](dn_fromsite);
    const SiteState* state_ia = &operator[](dn_tosite);

    //std::cout << " up_from = " << up_fromsite << "\n";
    //std::cout << " up_to   = " << up_tosite << "\n";
    //std::cout << " dn_from = " << dn_fromsite << "\n";
    //std::cout << " dn_to   = " << dn_tosite << "\n\n";
    // non-zero contribution from term only if followings
    if (!state_j->have_upspin()) continue;
    if (state_i->have_upspin() && up_fromsite!=up_tosite) continue;
    if (!state_jb->have_dnspin()) continue;
    if (state_ia->have_dnspin() && dn_fromsite!=dn_tosite) continue;

    unsigned upspin, dnspin;
    int delta_nd = 0;
    amplitude_t det_ratio1, det_ratio2;

    // first hop the up-spin
    upspin = state_j->upspin_id();
    if (up_fromsite == up_tosite) {
      det_ratio1 = amplitude_t(1.0);
    }
    else {
      wf.get_amplitudes(psi_row, up_tosite, dnspin_sites());
      det_ratio1 = psi_row.cwiseProduct(psi_inv.col(upspin)).sum();
    }

    // next hop the dn-spin
    dnspin = state_jb->dnspin_id();
    if (dn_fromsite == dn_tosite) {
      det_ratio2 = amplitude_t(1.0);
    }
    else {
      wf.get_amplitudes(psi_col, upspin_sites(), dn_tosite);
      // since one upspin have moved
      wf.get_amplitudes(psi_col(upspin), up_tosite, dn_tosite);
      // updated 'dnspin'-th row of psi_inv
      amplitude_t ratio_inv = amplitude_t(1.0)/det_ratio1;
      // elements other than 'upspin'-th
      for (unsigned i=0; i<upspin; ++i) {
        amplitude_t beta = ratio_inv*psi_row.cwiseProduct(psi_inv.col(i)).sum();
        inv_row(i) = psi_inv(dnspin,i) - beta * psi_inv(dnspin,upspin);
      }
      for (unsigned i=upspin+1; i<num_upspins_; ++i) {
        amplitude_t beta = ratio_inv*psi_row.cwiseProduct(psi_inv.col(i)).sum();
        inv_row(i) = psi_inv(dnspin,i) - beta * psi_inv(dnspin,upspin);
      }
      inv_row(upspin) = ratio_inv * psi_inv(dnspin,upspin);
      // ratio for the dnspin hop
      det_ratio2 = psi_col.cwiseProduct(inv_row).sum();
    }
    // net ratio for up & dn spin hop
    amplitude_t det_ratio = ampl_part(std::conj(det_ratio1*det_ratio2));

    if (pj.have_gutzwiller()) {
      // change in double occupancy for up-spin hop
      delta_nd = 0;
      if (up_fromsite != up_tosite) {
        delta_nd = state_i->count(); // must be 0 or one
        if (state_j->count()==2) delta_nd--;
      }
      // for dn-spin hop
      if (dn_fromsite != dn_tosite) {
        delta_nd += state_ia->count(); 
        if (state_jb->count()==2) delta_nd--;
      }
      det_ratio *= pj.gw_ratio(delta_nd);
    }
    // contribution from this term
    net_ratio += det_ratio;
    //std::cout << " det_ratio1 = " << det_ratio1 << "\n";
    //std::cout << " det_ratio2 = " << det_ratio2 << "\n";
    //std::cout << " delta_nd = " << delta_nd << "\n";
    //getchar();
  }

  int bc_phase = bphase_i * bphase_j;
  return 0.5 * bc_phase * net_ratio;
}

void SysConfig::get_grad_logpsi(RealVector& grad_logpsi) const
{
  // grad_logpsi wrt pj parameters
  unsigned p = pj.varparms().size();
  for (unsigned n=0; n<p; ++n) {
    if (pj.varparms()[n].name()=="gfactor") {
      double g = pj.varparms()[n].value();
      grad_logpsi(n) = static_cast<double>(dblocc_count())/g;
    }
    else {
      throw std::range_error("SysConfig::get_grad_logpsi: this pj parameter not implemented\n");
    }
  }
  // grad_logpsi wrt wf parameters
  for (unsigned n=0; n<wf.varparms().size(); ++n) {
    wf.get_gradients(psi_grad,n,upspin_sites(),dnspin_sites());
    grad_logpsi(p+n) = std::real(psi_grad.cwiseProduct(psi_inv.transpose()).sum());
  }
}

double SysConfig::accept_ratio(void)
{
  // acceptance ratio wrt particle number
  return static_cast<double>(last_accepted_moves_)/
         static_cast<double>(num_upspins_+num_dnspins_); 
  //return static_cast<double>(last_accepted_moves_)/
  //       static_cast<double>(last_proposed_moves_); 
}

void SysConfig::reset_accept_ratio(void)
{
  last_proposed_moves_ = 0;
  last_accepted_moves_ = 0;
}

void SysConfig::print_stats(std::ostream& os) const
{
  long proposed_hops = num_proposed_moves_[move_t::uphop]
                         + num_proposed_moves_[move_t::dnhop];
  long proposed_exch = num_proposed_moves_[move_t::exch];
  long accepted_hops = num_accepted_moves_[move_t::uphop] 
                     + num_accepted_moves_[move_t::dnhop];
  long accepted_exch = num_accepted_moves_[move_t::exch];
  double accept_ratio = 100.0*double(accepted_hops+accepted_exch)/(proposed_hops+proposed_exch);
  double hop_ratio = double(100.0*accepted_hops)/(proposed_hops);
  double exch_ratio = double(100.0*accepted_exch)/(proposed_exch);
  os << "--------------------------------------\n";
  os << " total mcsteps = " << num_updates_ <<"\n";
  os << " total accepted moves = " << (accepted_hops+accepted_exch)<<"\n";
  os << " acceptance ratio = " << accept_ratio << " %\n";
  os << " hopping = " << hop_ratio << " %\n";
  os << " exchange = " << exch_ratio << " %\n";
  os << "--------------------------------------\n";
}


} // end namespace vmc


