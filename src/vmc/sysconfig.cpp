/*---------------------------------------------------------------------------
* Author: Amal Medhi
* Date:   2017-02-18 14:01:12
* Last Modified by:   Amal Medhi, amedhi@macbook
* Last Modified time: 2017-05-20 11:16:31
* Copyright (C) Amal Medhi, amedhi@iisertvm.ac.in
*----------------------------------------------------------------------------*/
#include "./sysconfig.h"
#include <boost/algorithm/string.hpp>
#include <Eigen/SVD>

//#define HAVE_DETERMINANTAL_PART
#define MACHINE_ON

namespace vmc {

SysConfig::SysConfig(const input::Parameters& inputs, 
  const lattice::Lattice& lattice, const model::Hamiltonian& model)
  : fock_basis_(lattice.num_sites(), model.double_occupancy())
  , pj_(lattice,inputs)
  , wf_(lattice, inputs, model)
  , nqs_(lattice, inputs)
  , num_sites_(lattice.num_sites())
{
#ifdef HAVE_DETERMINANTAL_PART
  if (wf_.name() == "IDENTITY") have_mf_part_ = false;
  else have_mf_part_ = true;
#else
  have_mf_part_ = false;
#endif

  // variational parameters
  //nqs_.init_parameters(fock_basis_.rng(), 0.005);
  //num_net_parms_ = ffnet_.num_params();
  num_net_parms_ = nqs_.num_params();
  num_pj_parms_ = pj_.varparms().size();
  num_wf_parms_ = wf_.varparms().size();
  num_varparms_ = (num_net_parms_+num_pj_parms_+num_wf_parms_);
  vparm_names_.resize(num_varparms_);
  vparm_lbound_.resize(num_varparms_);
  vparm_ubound_.resize(num_varparms_);
  // names
  //ffnet_.get_parm_names(vparm_names_,0);
  nqs_.get_parm_names(vparm_names_,0);
  pj_.get_vparm_names(vparm_names_,num_net_parms_);
  wf_.get_vparm_names(vparm_names_,num_net_parms_+num_pj_parms_);
  // values are not static and may change
  // bounds
  //ffnet_.get_parm_lbound(vparm_lbound_,0);
  pj_.get_vparm_lbound(vparm_lbound_,num_net_parms_);
  wf_.get_vparm_lbound(vparm_lbound_,num_net_parms_+num_pj_parms_);
  //ffnet_.get_parm_ubound(vparm_ubound_,0);
  pj_.get_vparm_ubound(vparm_ubound_,num_net_parms_);
  wf_.get_vparm_ubound(vparm_ubound_,num_net_parms_+num_pj_parms_);
}

int SysConfig::init_files(const std::string& prefix, const input::Parameters& inputs)
{
  // variational parameters prefix folder
  int nowarn;
  /*
  save_path_ = inputs.set_value("parms_save_path", "", nowarn);
  boost::algorithm::trim(save_path_);
  if (save_path_.size()==0) save_path_ = "./";
  if (save_path_.back()!='/') save_path_ += "/";
  save_path_ += "vparms";
  nqs_.init_parameter_file(save_path_);
  */
  save_path_ = prefix+"/vparms";
  nqs_.init_parameter_file(save_path_);

  load_path_ = inputs.set_value("parms_load_path", "", nowarn);
  if (nowarn == 0) {
    boost::algorithm::trim(load_path_);
    if (load_path_.size()==0) load_path_ = "./";
    if (load_path_.back()!='/') load_path_ += "/";
    load_path_ += "vparms";
  }
  else {
    load_path_ = prefix+"/vparms";
  }
  // load from file option
  load_parms_from_file_ = inputs.set_value("load_parms_from_file",false);

  return 0;
}

const var::parm_vector& SysConfig::vparm_values(void) 
{
  // values as 'var::parm_vector'
  vparm_values_.resize(num_varparms_);
  //ffnet_.get_parm_values(vparm_values_,0);
  nqs_.get_parm_values(vparm_values_,0);
  pj_.get_vparm_values(vparm_values_,num_net_parms_);
  wf_.get_vparm_values(vparm_values_,num_net_parms_+num_pj_parms_);
  return vparm_values_;
}

const std::vector<double>& SysConfig::vparm_vector(void) 
{
  // values as 'std::double'
  vparm_vector_.resize(num_varparms_);
  //ffnet_.get_parm_vector(vparm_vector_,0);
  //nqs_.get_parm_vector(vparm_vector_,0);
  pj_.get_vparm_vector(vparm_vector_,num_net_parms_);
  wf_.get_vparm_vector(vparm_vector_,num_net_parms_+num_pj_parms_);
  return vparm_vector_;
}

std::string SysConfig::info_str(void) const
{
  std::ostringstream info;
  info << wf_.info_str();
  info.precision(6);
  info.setf(std::ios_base::fixed);
  /*info << "# Variational parameters:\n";
  for (const auto& v : wf.varparms()) 
    info << "# " << v.name() << " = " << v.value() << "\n";
  for (const auto& v : pj.varparms()) 
    info << "# " << v.name() << " = " << v.value() << "\n";
  */
  return info.str();
}

int SysConfig::build(const lattice::Lattice& lattice, const input::Parameters& inputs,
    const bool& with_gradient)
{
  if (num_sites_==0) return -1;
  pj_.update(inputs);
#ifdef HAVE_DETERMINANTAL_PART
  wf_.compute(lattice, inputs, with_gradient);
#endif
  nqs_.init_parameters(fock_basis_.rng(), 0.1);
  if (load_parms_from_file_) {
    nqs_.load_parameters(load_path_);
  }
  //---------TEST-----------
  int xn = inputs.set_value("xn", 0);
  double xp = inputs.set_value("xp", 0.0);
  nqs_.update_parameter(xn, xp);



  init_config();
  return 0;
}

int SysConfig::build(const lattice::Lattice& lattice, const var::parm_vector& pvector,
  const bool& need_psi_grad)
{
  if (num_sites_==0) return -1;
  int start_pos = 0;
  //std::cout << "pvector = " << pvector.transpose() << "\n"; getchar();
  nqs_.update_parameters(pvector, start_pos);

  start_pos += num_net_parms_;
  pj_.update(pvector,start_pos);
  start_pos += num_pj_parms_;
#ifdef HAVE_DETERMINANTAL_PART
  wf_.compute(lattice, pvector, start_pos, need_psi_grad);
#endif
  init_config();
  return 0;
}

// rebuild for new lattice boundary twist
int SysConfig::rebuild(const lattice::Lattice& lattice)
{
#ifdef HAVE_DETERMINANTAL_PART
  wf_.recompute(lattice);
#endif
  //nqs_.init_parameters(fock_basis_.rng(), 0.005);
  if (load_parms_from_file_) {
    nqs_.load_parameters(load_path_);
  }
  init_config();
  return 0;
}

int SysConfig::save_parameters(const var::parm_vector& pvector)
{
  int start_pos = 0;
  nqs_.update_parameters(pvector, start_pos);
  nqs_.save_parameters();
  return 0;
}

int SysConfig::init_config(void)
{
  num_upspins_ = wf_.num_upspins();
  num_dnspins_ = wf_.num_dnspins();
  if (num_upspins_==0 && num_dnspins_==0) return -1;
  if (num_upspins_ != num_dnspins_) 
    throw std::range_error("*SysConfig::init_config: unequal UP & DN spin case not implemented");
  fock_basis_.init_spins(num_upspins_,num_dnspins_);

#ifdef HAVE_DETERMINANTAL_PART
  if (have_mf_part_) {
    psi_mat_.resize(num_upspins_, num_dnspins_);
    psi_inv_.resize(num_dnspins_, num_upspins_);
    // try for a well condictioned amplitude matrix
    fock_basis_.set_random();
    int num_attempt = 0;
    while (true) {
      wf_.get_amplitudes(psi_mat_,fock_basis_.upspin_sites(),fock_basis_.dnspin_sites());
      // reciprocal conditioning number
      Eigen::JacobiSVD<Matrix> svd(psi_mat_);
      // reciprocal cond. num = smallest eigenval/largest eigen val
      double rcond = svd.singularValues()(svd.singularValues().size()-1)/svd.singularValues()(0);
      if (std::isnan(rcond)) rcond = 0.0; 
      //std::cout << "rcondition number = "<< rcond << "\n";
      if (rcond>1.0E-15) break;
      // try new basis state
      fock_basis_.set_random();
      if (++num_attempt > 1000) {
        throw std::underflow_error("*SysConfig::init: configuration wave function ill conditioned.");
      }
    }
    // amplitude matrix invers
    psi_inv_ = psi_mat_.inverse();
  }
#endif
  // run parameters
  //ffnet_.update_state(fock_basis_.state());
  //ffn_psi_ = ffnet_.output();
  nqs_.update_state(fock_basis_.state());
  nqs_psi_ = nqs_.output();
  nqs_sign_ = 1;

  set_run_parameters();
  return 0;
}

int SysConfig::set_run_parameters(void)
{
  num_updates_ = 0;
  num_iterations_ = 0;
  refresh_cycle_ = 100;
  // number of moves per mcstep
  int n_up = static_cast<int>(num_upspins_);
  int n_dn = static_cast<int>(num_dnspins_);
  if (fock_basis_.double_occupancy()) {
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
  psi_row_.resize(num_dnspins_);
  psi_col_.resize(num_upspins_);
  inv_row_.resize(num_upspins_);
  psi_grad_.resize(num_upspins_,num_dnspins_);
  return 0;
}

int SysConfig::update_state(void)
{
  //for (int n=0; n<1; ++n) do_upspin_hop();
  //std::cout<<"hop sign = "<<fock_basis_.op_sign()<<"\n";
  for (int n=0; n<num_uphop_moves_; ++n) do_upspin_hop();
  for (int n=0; n<num_dnhop_moves_; ++n) do_dnspin_hop();
  for (int n=0; n<num_exchange_moves_; ++n) do_spin_exchange();

  
  /*
  if (true) {
  //if (false) {
    auto psi = psi_mat_.determinant();
    std::cout<<std::scientific<<std::uppercase<<std::setprecision(6)<<std::right;
    //std::cout<<std::setw(15)<<nqs_psi_<<std::setw(15)<<psi.real()*nqs_sign_
    //   <<std::setw(15)<<psi.imag()*nqs_sign_<< " ratio = "<<double(nqs_sign_)*psi/nqs_psi_<<"\n";
    std::cout<<fock_basis_<<" : ampl = ";
    std::cout<<std::setw(15)<<psi.real()<<", "<<std::setw(15)<<psi.imag()<<"\n";

    // Calculate NQS wf for states generated by FS wf
    nqs_.update_state(fock_basis_.state());
    nqs_psi_ = nqs_.output();
    std::cout<<fock_basis_<<" : ampl = ";
    std::cout<<std::setw(15)<<nqs_psi_.real()<<", "<<std::setw(15)<<nqs_psi_.imag()<<"\n";
    std::cout<<"FS/NQS = "<<std::setw(15)<<psi/nqs_psi_<<"\n";
    std::cout<<"\n";
  }
  */

  //Matrix psi_ord(num_upspins_,num_dnspins_);
  //std::vector<int> upsites = fock_basis_.upspin_sites();
  //std::vector<int> dnsites = fock_basis_.dnspin_sites();
  //std::sort(upsites.begin(),upsites.end());
  //std::sort(dnsites.begin(),dnsites.end());
  //wf_.get_amplitudes(psi_ord,upsites,dnsites);
  //auto psi2 = psi_ord.determinant();
  //std::cout << "sign = " << nqs_sign_ << " --- " << std::real(psi/psi2) << "\n";


  // CHECK
  /* 
  auto psi = psi_mat_.determinant();
  std::cout << "Psi = " << psi << "\n";
  std::cout << "NQS = " << nqs_psi_ << "\n";
  std::cout << "ratio = " << psi.real()/nqs_psi_ << "\n";
  getchar();
  */

  num_updates_++;
  if (have_mf_part_) {
    num_iterations_++;
    if (num_iterations_ == refresh_cycle_) {
      psi_inv_ = psi_mat_.inverse();
      num_iterations_ = 0;
    }
  }
  return 0;
}

int SysConfig::do_upspin_hop(void)
{
  if (fock_basis_.gen_upspin_hop()) {
    num_proposed_moves_[move_t::uphop]++;
    last_proposed_moves_++;
    //std::cout << "\n state=" << fock_basis_.transpose() << "\n";
    amplitude_t psi = nqs_.get_new_output(fock_basis_.state(),fock_basis_.new_elems());
#ifdef MACHINE_ON
    amplitude_t psi_ratio = psi/nqs_psi_;
#else
    amplitude_t psi_ratio = 1.0;
#endif

#ifdef HAVE_DETERMINANTAL_PART
    double proj_ratio = pj_.gw_ratio(fock_basis_.delta_nd());
    psi_ratio *= proj_ratio;
    // determinantal part
    int upspin, to_site;
    amplitude_t det_ratio;
    if (have_mf_part_) {
      upspin = fock_basis_.which_upspin();
      to_site = fock_basis_.which_site();
      wf_.get_amplitudes(psi_row_,to_site,fock_basis_.dnspin_sites());
      det_ratio = psi_row_.cwiseProduct(psi_inv_.col(upspin)).sum();
      //----To just compare: SWITCH OFF next 2 lines----
      if (std::abs(det_ratio) < 1.0E-12) det_ratio = 0.0;
      psi_ratio *= det_ratio;
    }
#endif

    double transition_proby = std::norm(psi_ratio);
    if (fock_basis_.rng().random_real()<transition_proby) {
      num_accepted_moves_[move_t::uphop]++;
      last_accepted_moves_++;
      // upddate state
      fock_basis_.commit_last_move();
      nqs_.update_state(fock_basis_.state(),fock_basis_.new_elems());
      nqs_psi_ = nqs_.output();
      nqs_sign_ *= fock_basis_.op_sign();

#ifdef HAVE_DETERMINANTAL_PART
      if (have_mf_part_) {
        inv_update_upspin(upspin,psi_row_,det_ratio);
      }
#endif

    }
    else {
      fock_basis_.undo_last_move();
    }
  } 
  return 0;
}

int SysConfig::do_dnspin_hop(void)
{
  if (fock_basis_.gen_dnspin_hop()) {
    num_proposed_moves_[move_t::dnhop]++;
    last_proposed_moves_++;
    //std::cout << "\n state=" << fock_basis_.transpose() << "\n";
    amplitude_t psi = nqs_.get_new_output(fock_basis_.state(),fock_basis_.new_elems());
#ifdef MACHINE_ON
    amplitude_t psi_ratio = psi/nqs_psi_;
#else
    amplitude_t psi_ratio = 1.0;
#endif

#ifdef HAVE_DETERMINANTAL_PART
    double proj_ratio = pj_.gw_ratio(fock_basis_.delta_nd());
    psi_ratio *= proj_ratio;
    // determinantal part
    int dnspin, to_site;
    amplitude_t det_ratio;
    if (have_mf_part_) {
      dnspin = fock_basis_.which_dnspin();
      to_site = fock_basis_.which_site();
      wf_.get_amplitudes(psi_col_,fock_basis_.upspin_sites(),to_site);
      det_ratio = psi_col_.cwiseProduct(psi_inv_.row(dnspin)).sum();
      //----To just compare: SWITCH OFF next 2 lines----
      if (std::abs(det_ratio) < 1.0E-12) det_ratio = 0.0;
      psi_ratio *= det_ratio;
    }
#endif
    double transition_proby = std::norm(psi_ratio);
    if (fock_basis_.rng().random_real()<transition_proby) {
      num_accepted_moves_[move_t::dnhop]++;
      last_accepted_moves_++;
      // upddate state
      fock_basis_.commit_last_move();
      nqs_.update_state(fock_basis_.state(),fock_basis_.new_elems());
      nqs_psi_ = nqs_.output();
      nqs_sign_ *= fock_basis_.op_sign();
#ifdef HAVE_DETERMINANTAL_PART
      if (have_mf_part_) {
        inv_update_dnspin(dnspin,psi_col_,det_ratio);
      }
#endif
    }
    else {
      fock_basis_.undo_last_move();
    }
  } 
  return 0;
}

int SysConfig::do_spin_exchange(void)
{
  if (fock_basis_.gen_exchange_move()) {
    num_proposed_moves_[move_t::exch]++;
    last_proposed_moves_++;
    //std::cout << "\n state=" << fock_basis_.transpose() << "\n";
    amplitude_t psi = nqs_.get_new_output(fock_basis_.state(),fock_basis_.new_elems());
#ifdef MACHINE_ON
    amplitude_t psi_ratio = psi/nqs_psi_;
#else
    amplitude_t psi_ratio = 1.0;
#endif

#ifdef HAVE_DETERMINANTAL_PART
    int upspin, up_tosite, dnspin, dn_tosite;
    amplitude_t det_ratio1, det_ratio2;
    if (have_mf_part_) {
      upspin = fock_basis_.which_upspin();
      up_tosite = fock_basis_.which_upspin_site();
      // for upspin hop forward
      wf_.get_amplitudes(psi_row_, up_tosite, fock_basis_.dnspin_sites());
      det_ratio1 = psi_row_.cwiseProduct(psi_inv_.col(upspin)).sum();
      if (std::abs(det_ratio1) < 1.0E-12) {
        fock_basis_.undo_last_move();
        return 0; // for safety
      } 
      // now for dnspin hop backward
      dnspin = fock_basis_.which_dnspin();
      dn_tosite = fock_basis_.which_dnspin_site();
      // new col for this move
      wf_.get_amplitudes(psi_col_,fock_basis_.upspin_sites(), dn_tosite);
      // since the upspin should have moved
      wf_.get_amplitudes(psi_col_(upspin), up_tosite, dn_tosite);
      // updated 'dnspin'-th row of psi_inv
      amplitude_t ratio_inv = amplitude_t(1.0)/det_ratio1;
      // elements other than 'upspin'-th
      for (int i=0; i<upspin; ++i) {
        amplitude_t beta = ratio_inv*psi_row_.cwiseProduct(psi_inv_.col(i)).sum();
        inv_row_(i) = psi_inv_(dnspin,i) - beta * psi_inv_(dnspin,upspin);
      }
      for (int i=upspin+1; i<num_upspins_; ++i) {
        amplitude_t beta = ratio_inv*psi_row_.cwiseProduct(psi_inv_.col(i)).sum();
        inv_row_(i) = psi_inv_(dnspin,i) - beta * psi_inv_(dnspin,upspin);
      }
      inv_row_(upspin) = ratio_inv * psi_inv_(dnspin,upspin);
      // ratio for the dnspin hop
      det_ratio2 = psi_col_.cwiseProduct(inv_row_).sum();
      if (std::abs(det_ratio2) < dratio_cutoff()) {
        fock_basis_.undo_last_move();
        return 0; // for safety
      }
      //----To just compare: SWITCH OFF next 1 line----
      psi_ratio = psi_ratio*det_ratio1*det_ratio2;
    }
#endif
    double transition_proby = std::norm(psi_ratio);
    if (fock_basis_.rng().random_real()<transition_proby) {
      num_accepted_moves_[move_t::exch]++;
      last_accepted_moves_++;
      fock_basis_.commit_last_move();
      nqs_.update_state(fock_basis_.state(),fock_basis_.new_elems());
      nqs_psi_ = nqs_.output();
      nqs_sign_ *= fock_basis_.op_sign();
#ifdef HAVE_DETERMINANTAL_PART
      if (have_mf_part_) {
        inv_update_upspin(upspin,psi_row_,det_ratio1);
        inv_update_dnspin(dnspin,psi_col_,det_ratio2);
      }
#endif
    }
    else {
      fock_basis_.undo_last_move();
    }
  } 
  return 0;
}

int SysConfig::inv_update_upspin(const int& upspin, const ColVector& psi_row, 
  const amplitude_t& det_ratio)
{
  psi_mat_.row(upspin) = psi_row;
  amplitude_t ratio_inv = amplitude_t(1.0)/det_ratio;
  for (int i=0; i<upspin; ++i) {
    amplitude_t beta = ratio_inv*psi_row.cwiseProduct(psi_inv_.col(i)).sum();
    psi_inv_.col(i) -= beta * psi_inv_.col(upspin);
  }
  for (int i=upspin+1; i<num_upspins_; ++i) {
    amplitude_t beta = ratio_inv*psi_row.cwiseProduct(psi_inv_.col(i)).sum();
    psi_inv_.col(i) -= beta * psi_inv_.col(upspin);
  }
  psi_inv_.col(upspin) *= ratio_inv;
  return 0;
}

int SysConfig::inv_update_dnspin(const int& dnspin, const RowVector& psi_col, 
  const amplitude_t& det_ratio)
{
  psi_mat_.col(dnspin) = psi_col;
  amplitude_t ratio_inv = amplitude_t(1.0)/det_ratio;
  for (int i=0; i<dnspin; ++i) {
    amplitude_t beta = ratio_inv*psi_col_.cwiseProduct(psi_inv_.row(i)).sum();
    psi_inv_.row(i) -= beta * psi_inv_.row(dnspin);
  }
  for (int i=dnspin+1; i<num_dnspins_; ++i) {
    amplitude_t beta = ratio_inv*psi_col_.cwiseProduct(psi_inv_.row(i)).sum();
    psi_inv_.row(i) -= beta * psi_inv_.row(dnspin);
  }
  psi_inv_.row(dnspin) *= ratio_inv;
  return 0;
}

int SysConfig::apply_niup_nidn(const int& site_i) const
{
  return fock_basis_.op_ni_updn(site_i);
}

int SysConfig::apply_ni_dblon(const int& site_i) const
{
  return fock_basis_.op_ni_dblon(site_i);
}

int SysConfig::apply_ni_holon(const int& site_i) const
{
  return fock_basis_.op_ni_holon(site_i);
}

int SysConfig::apply(const model::op::quantum_op& qn_op, const int& site_i) const
{
  switch (qn_op.id()) {
    case model::op_id::ni_sigma:
      return fock_basis_.op_ni_up(site_i)+fock_basis_.op_ni_dn(site_i);
    case model::op_id::ni_up:
      return fock_basis_.op_ni_up(site_i);
    case model::op_id::ni_dn:
      return fock_basis_.op_ni_dn(site_i);
    case model::op_id::niup_nidn:
      return fock_basis_.op_ni_updn(site_i);
    case model::op_id::Sz:
      return fock_basis_.op_Sz(site_i); 
    default: 
      throw std::range_error("SysConfig::apply: undefined site operator");
  }
}

amplitude_t SysConfig::apply(const model::op::quantum_op& qn_op, 
  const int& site_i, const int& site_j, const int& bc_state, 
  const std::complex<double>& bc_phase) const
{
  amplitude_t term(0); 
  switch (qn_op.id()) {
    case model::op_id::cdagc_sigma:
      term = apply_cdagc_up(site_i,site_j,bc_state,bc_phase);
      term+= apply_cdagc_dn(site_i,site_j,bc_state,bc_phase);
      break;
    case model::op_id::cdagc2_sigma:
      term = apply_cdagc2_up(site_i,site_j,bc_state,bc_phase);
      term+= apply_cdagc2_dn(site_i,site_j,bc_state,bc_phase);
      break;
    case model::op_id::sisj_plus:
      term = apply_sisj_plus(site_i,site_j); 
      break;
    default: 
      throw std::range_error("SysConfig::apply: undefined bond operator.");
  }
  return term;
}


amplitude_t SysConfig::apply_cdagc_up(const int& i, const int& j,
  const int& bc_state, const std::complex<double>& bc_phase) const
{
  if (i == j) return ampl_part(fock_basis_.op_ni_up(i));
  if (fock_basis_.op_cdagc_up(i,j)) {
    int sign = fock_basis_.op_sign();
    amplitude_t psi = nqs_.get_new_output(fock_basis_.state());
#ifdef MACHINE_ON
    amplitude_t psi_ratio = psi/nqs_psi_;
    //std::cout << psi << "\n";
    //sign *= nqs_sign_;
#else
    amplitude_t psi_ratio = 1.0;
#endif

    /*
    if (bc_state < 0) {
      std::cout << "sign = " << sign << "\n"; 
      std::cout << "bc_phase = "<<std::real(bc_phase)<<"\n";
      std::cout << "state = "<<fock_basis_<<"\n";
      getchar();
    }*/

    //----To just compare: SWITCH OFF ifdef block----
#ifdef HAVE_DETERMINANTAL_PART
    double proj_ratio = pj_.gw_ratio(fock_basis_.delta_nd());
    if (have_mf_part_) {
      int upspin = fock_basis_.which_upspin();
      int to_site = fock_basis_.which_site();
      wf_.get_amplitudes(psi_row_,to_site,fock_basis_.dnspin_sites());
      amplitude_t det_ratio = psi_row_.cwiseProduct(psi_inv_.col(upspin)).sum();
      psi_ratio *= proj_ratio*det_ratio;
    }
#endif
    fock_basis_.undo_last_move(); 
    /* Necessary to 'undo', as next measurement could be 
      'site diagonal' where no 'undo' is done */
    if (bc_state == -1) {
      // it's a boundary bond
      //return psi_ratio*ampl_part(bc_phase);
      return psi_ratio*std::real(bc_phase);
    } 
    else {
      return psi_ratio;
    }
  }
  else return amplitude_t(0.0);
}

amplitude_t SysConfig::apply_cdagc2_up(const int& i, const int& j,
  const int& bc_state, const std::complex<double>& bc_phase) const
{
  if (i == j) return ampl_part(fock_basis_.op_ni_up(i));
  if (fock_basis_.op_cdagc2_up(i,j)) {
    int sign = fock_basis_.op_sign();
    amplitude_t psi = nqs_.get_new_output(fock_basis_.state());
#ifdef MACHINE_ON
    amplitude_t psi_ratio = psi/nqs_psi_;
    //std::cout << psi << "\n";
    //sign *= nqs_sign_;
#else
    amplitude_t psi_ratio = 1.0;
#endif

    /*
    if (bc_state < 0) {
      std::cout << "sign = " << sign << "\n"; 
      std::cout << "bc_phase = "<<std::real(bc_phase)<<"\n";
      std::cout << "state = "<<fock_basis_<<"\n";
      getchar();
    }*/
    //----To just compare: SWITCH OFF ifdef block----
#ifdef HAVE_DETERMINANTAL_PART
    double proj_ratio = pj_.gw_ratio(fock_basis_.delta_nd());
    if (have_mf_part_) {
      int upspin = fock_basis_.which_upspin();
      int to_site = fock_basis_.which_site();
      wf_.get_amplitudes(psi_row_,to_site,fock_basis_.dnspin_sites());
      amplitude_t det_ratio = psi_row_.cwiseProduct(psi_inv_.col(upspin)).sum();
      psi_ratio *= proj_ratio*det_ratio;
    }
#endif
    fock_basis_.undo_last_move(); 
    /* Necessary to 'undo', as next measurement could be 
      'site diagonal' where no 'undo' is done */
    if (bc_state == -1) {
      // it's a boundary bond
      //return psi_ratio*ampl_part(bc_phase);
      return psi_ratio*std::real(bc_phase);
    } 
    else {
      return psi_ratio;
    }
  }
  else return amplitude_t(0.0);
}

amplitude_t SysConfig::apply_cdagc_dn(const int& i, const int& j,
  const int& bc_state, const std::complex<double>& bc_phase) const
{
  if (i == j) return ampl_part(fock_basis_.op_ni_dn(i));
  if (fock_basis_.op_cdagc_dn(i,j)) {
    int sign = fock_basis_.op_sign();
    amplitude_t psi = nqs_.get_new_output(fock_basis_.state());
#ifdef MACHINE_ON
    amplitude_t psi_ratio = psi/nqs_psi_;
    //sign *= nqs_sign_;
#else
    amplitude_t psi_ratio = 1.0;
#endif

    //----To just compare: SWITCH OFF ifdef block----
#ifdef HAVE_DETERMINANTAL_PART
    double proj_ratio = pj_.gw_ratio(fock_basis_.delta_nd());
    if (have_mf_part_) {
      int dnspin = fock_basis_.which_dnspin();
      int to_site = fock_basis_.which_site();
      wf_.get_amplitudes(psi_col_,fock_basis_.upspin_sites(),to_site);
      amplitude_t det_ratio = psi_col_.cwiseProduct(psi_inv_.row(dnspin)).sum();
      //psi_ratio *= amplitude_t(proj_ratio*bc_phase)*det_ratio;
      psi_ratio *= proj_ratio*det_ratio;
    }
#endif
    fock_basis_.undo_last_move(); 
    /* Necessary to 'undo', as next measurement could be 
      'site diagonal' where no 'undo' is done */
    if (bc_state == -1) {
      // it's a boundary bond
      //return psi_ratio*ampl_part(bc_phase);
      return psi_ratio*std::real(bc_phase);
    } 
    else {
      return psi_ratio;
    }
  }
  else return amplitude_t(0.0);
}

amplitude_t SysConfig::apply_cdagc2_dn(const int& i, const int& j,
  const int& bc_state, const std::complex<double>& bc_phase) const
{
  if (i == j) return ampl_part(fock_basis_.op_ni_dn(i));
  if (fock_basis_.op_cdagc2_dn(i,j)) {
    int sign = fock_basis_.op_sign();
    amplitude_t psi = nqs_.get_new_output(fock_basis_.state());
#ifdef MACHINE_ON
    amplitude_t psi_ratio = psi/nqs_psi_;
    //sign *= nqs_sign_;
#else
    amplitude_t psi_ratio = 1.0;
#endif


    //----To just compare: SWITCH OFF ifdef block----
#ifdef HAVE_DETERMINANTAL_PART
    double proj_ratio = pj_.gw_ratio(fock_basis_.delta_nd());
    if (have_mf_part_) {
      int dnspin = fock_basis_.which_dnspin();
      int to_site = fock_basis_.which_site();
      wf_.get_amplitudes(psi_col_,fock_basis_.upspin_sites(),to_site);
      amplitude_t det_ratio = psi_col_.cwiseProduct(psi_inv_.row(dnspin)).sum();
      //psi_ratio *= amplitude_t(proj_ratio*bc_phase)*det_ratio;
      psi_ratio *= proj_ratio*det_ratio;
    }
#endif
    fock_basis_.undo_last_move(); 
    /* Necessary to 'undo', as next measurement could be 
      'site diagonal' where no 'undo' is done */
    if (bc_state == -1) {
      // it's a boundary bond
      //return psi_ratio*ampl_part(bc_phase);
      return psi_ratio*std::real(bc_phase);
    } 
    else {
      return psi_ratio;
    }
  }
  else return amplitude_t(0.0);
}

amplitude_t SysConfig::apply_sisj_plus(const int& i, const int& j) const
{
/* It evaluates the following operator:
 !   O = (S_i.S_j - (n_i n_j)/4)
 ! The operator can be cast in the form,
 !   O = O_{ud} + O_{du}
 ! where,
 !   O_{ud} = 1/2*(- c^{\dag}_{j\up}c_{i\up} c^{\dag}_{i\dn}c_{j\dn}
 !                 - n_{i\up} n_{j_dn})
 ! O_{du} is obtained from O_{ud} by interchanging spin indices. */

#ifdef OLD_STAFF
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

  int upspin, up_tosite;
  int dnspin, dn_tosite;
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
  wf_.get_amplitudes(psi_row_, up_tosite, dnspin_sites());
  amplitude_t det_ratio1 = psi_row_.cwiseProduct(psi_inv_.col(upspin)).sum();
  // now for dnspin hop 
  wf_.get_amplitudes(psi_col_, upspin_sites(), dn_tosite);
  // since the upspin should have moved
  wf_.get_amplitudes(psi_col_(upspin), up_tosite, dn_tosite);
  // updated 'dnspin'-th row of psi_inv
  amplitude_t ratio_inv = amplitude_t(1.0)/det_ratio1;

  // for safety: if 'det_ratio1 == 0', result is zero
  if (std::isinf(std::abs(ratio_inv))) {
    return amplitude_t(ninj_term);
  }

  // elements other than 'upspin'-th
  for (int i=0; i<upspin; ++i) {
    amplitude_t beta = ratio_inv*psi_row_.cwiseProduct(psi_inv_.col(i)).sum();
    inv_row_(i) = psi_inv_(dnspin,i) - beta * psi_inv_(dnspin,upspin);
  }
  for (int i=upspin+1; i<num_upspins_; ++i) {
    amplitude_t beta = ratio_inv*psi_row_.cwiseProduct(psi_inv_.col(i)).sum();
    inv_row_(i) = psi_inv_(dnspin,i) - beta * psi_inv_(dnspin,upspin);
  }
  inv_row_(upspin) = ratio_inv * psi_inv_(dnspin,upspin);
  // ratio for the dnspin hop
  amplitude_t det_ratio2 = psi_col_.cwiseProduct(inv_row_).sum();
  amplitude_t det_ratio = ampl_part(std::conj(det_ratio1*det_ratio2));
  /*
  if (std::isnan(det_ratio)) {
    std::cout << std::scientific<< det_ratio1 << "\n\n";
    std::cout << std::scientific<< ratio_inv << "\n\n";
    std::cout << std::scientific<< det_ratio2 << "\n\n";
    std::cout << "NaN detected\n"; getchar();
  }*/
  return -0.5 * det_ratio + amplitude_t(ninj_term);
#endif
  return 0.0;
}

amplitude_t SysConfig::apply_bondsinglet_hop(const int& fr_site_i, 
  const int& fr_site_ia, const int& to_site_j, const int& to_site_jb) const
{
/*----------------------------------------------------------------------
* Evaluates the following operator:
* Denoting: 
*          c^{dag}_{i\sigma} = d_{i\sigma}
*          c_{i\sigma}       = c_{i\sigma}
*          1/\sqrt{2}        = 1sqrt2 
*          1/2               = half 
*
* F_{ab}(i,j) = 1sqrt2 x (d_{i\up}d_{i+a\dn} - d_{i\dn}d_{i+a\up}) x
*               1sqrt2 x (c_{i\dn}c_{i+b\up} - c_{i\up}c_{i+b\dn}) 
*
* It can be written as sum of 4 terms:
* 
* F_{ab}(i,j) = 0.5 x [ (d_{i+a\dn}c_{j+b\dn) (d_{i\up}c_{j\up}) 
*                     + (d_{i+a\dn}c_{j\dn})  (d_{i\up}c_{j+b\up})
*                     + (d_{i\dn}c_{j+b\dn})  (d_{i+a\up}c_{j\up}) 
*                     + (d_{i\dn}c_{j\dn})    (d_{i+a\up}c_{j+b\up}) ]
*
* Each term describes a 'UP'-spin hop followed by a 'DOWN'-spin hop
*
* Assumption: No hopping crosses boundary (no need to consider BC twists)
*-----------------------------------------------------------------------*/
  return 0.0;
}

amplitude_t SysConfig::apply_sitepair_hop(const int& fr_site, const int& to_site) const
{
/*----------------------------------------------------------------------
* Evaluates the following operator.
* Denote: 
*          c^{dag}_{i\sigma} = d_{i\sigma}
*          c_{i\sigma}       = c_{i\sigma}
*
* F_{ab}(i,j) = (d_{i\up}d_{i\dn}) (c_{j\dn}c_{j\up}
*             = (d_{i\dn}c_{j\dn}) (d_{i\up}c_{j\up)
*
* The term describes a 'UP'-spin hop followed by a 'DOWN'-spin hop 
* between same sites
*-----------------------------------------------------------------------*/
  return 0.0;
}

void SysConfig::get_grad_logpsi(Vector& grad_logpsi) const
{
  // grad_logpsi wrt nnet parameters
  Vector grad(num_net_parms_);
  //std::cout << "getting grad" << "\n"; getchar();
  nqs_.get_log_gradient(grad_logpsi, 0);

  /*
  std::cout << grad_logpsi.transpose() << "\n"; getchar();
  // grad_logpsi wrt pj parameters
  for (int n=0; n<num_pj_parms_; ++n) {
    if (pj_.varparms()[n].name()=="gfactor") {
      double g = pj_.varparms()[n].value();
      grad_logpsi(num_net_parms_+n) = static_cast<double>(dblocc_count())/g;
    }
    else {
      throw std::range_error("SysConfig::get_grad_logpsi: this pj parameter not implemented\n");
    }
  }
  // grad_logpsi wrt wf parameters
  //auto grad = ffnet_.get_gradient()/ffn_psi_;
  int p = num_net_parms_+num_pj_parms_;
  for (int n=0; n<num_wf_parms_; ++n) {
    wf_.get_gradients(psi_grad_,n,fock_basis_.upspin_sites(),fock_basis_.dnspin_sites());
    grad_logpsi(p+n) = std::real(psi_grad_.cwiseProduct(psi_inv_.transpose()).sum());
    //grad_logpsi(p+n) = grad(n);
  }
  */
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
  double accept_ratio = (proposed_hops+proposed_exch) > 0
    ? 100.0*double(accepted_hops+accepted_exch)/(proposed_hops+proposed_exch) : 0.0;
  double hop_ratio = proposed_hops > 0 
    ? double(100.0*accepted_hops)/(proposed_hops) : 0.0;
  double exch_ratio = proposed_exch > 0
    ? double(100.0*accepted_exch)/(proposed_exch) : 0.0;
  os << "--------------------------------------\n";
  os << " total mcsteps = " << num_updates_ <<"\n";
  os << " total accepted moves = " << (accepted_hops+accepted_exch)<<"\n";
  os << " acceptance ratio = " << accept_ratio << " %\n";
  os << " hopping = " << hop_ratio << " %\n";
  os << " exchange = " << exch_ratio << " %\n";
  os << "--------------------------------------\n";
}


} // end namespace vmc


