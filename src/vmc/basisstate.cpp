/*---------------------------------------------------------------------------
* Author: Amal Medhi
* Date:   2017-02-13 10:20:28
* Last Modified by:   Amal Medhi, amedhi@macbook
* Last Modified time: 2017-02-20 04:54:18
* Copyright (C) Amal Medhi, amedhi@iisertvm.ac.in
*----------------------------------------------------------------------------*/
#include "basisstate.h"
#include <stdexcept>
#include <algorithm>

namespace vmc {

void FockBasis::init(const int& num_sites, const bool& allow_dbl) 
{
  num_sites_ = num_sites;
  num_states_ = 2*num_sites_;
  state_.resize(num_states_);
  state_.setZero();
  //ssign_ = 1;
  spin_id_.resize(num_states_);
  spin_id_.setConstant(-1); 
  double_occupancy_ = allow_dbl;
  up_states_.clear();
  dn_states_.clear();
  uphole_states_.clear();
  dnhole_states_.clear();
  proposed_move_ = move_t::null;
  // rng site generator
  if (num_sites_>0) rng_.set_site_generator(0,num_sites_-1);
}

void FockBasis::init_spins(const int& num_upspins, const int& num_dnspins)
{
  num_upspins_ = num_upspins;
  num_dnspins_ = num_dnspins;
  if (num_upspins_>num_sites_ || num_dnspins_>num_sites_)
    throw std::range_error("* FockBasis::init_spins: spin number exceeds capacity");
  if (!double_occupancy_ && (num_upspins_+num_dnspins_)>num_sites_)
    throw std::range_error("* FockBasis::init_spins: spin number exceeds capacity");
  num_upholes_ = num_sites_ - num_upspins;
  num_dnholes_ = num_sites_ - num_dnspins;
  // resizing
  up_states_.resize(num_upspins_);
  dn_states_.resize(num_dnspins_);
  dnspin_sites_.resize(num_dnspins_);
  uphole_states_.resize(num_upholes_);
  dnhole_states_.resize(num_dnholes_);
  // random generator
  if (num_upspins_>0) rng_.set_upspin_generator(0,num_upspins_-1);
  else rng_.set_upspin_generator(0,0);
  if (num_dnspins_>0) rng_.set_dnspin_generator(0,num_dnspins_-1);
  else rng_.set_dnspin_generator(0,0);
  if (num_upholes_>0) rng_.set_uphole_generator(0,num_upholes_-1);
  else rng_.set_uphole_generator(0,0);
  if (num_dnholes_>0) rng_.set_dnhole_generator(0,num_dnholes_-1);
  else rng_.set_dnhole_generator(0,0);
  // random initial configuration
  set_random();
}

void FockBasis::set_random(void)
{
  proposed_move_ = move_t::null;
  //ssign_ = 1;
  state_.setZero();
  std::vector<int> all_up_states(num_sites_);
  for (int i=0; i<num_sites_; ++i) all_up_states[i] = i;
  std::shuffle(all_up_states.begin(),all_up_states.end(),rng_);
  for (int i=0; i<num_upspins_; ++i) {
    int state = all_up_states[i];
    state_[state] = 1;
    spin_id_[state] = i;
    up_states_[i] = state;
  }
  int j=0;
  for (int i=num_upspins_; i<num_sites_; ++i) {
    uphole_states_[j++] = all_up_states[i];
  }

  // DN spins & holes
  if (double_occupancy_) {
    std::vector<int> all_dn_states(num_sites_);
    for (int i=0; i<num_sites_; ++i) all_dn_states[i] = num_sites_+i;
    std::shuffle(all_dn_states.begin(),all_dn_states.end(),rng_);
    for (int i=0; i<num_dnspins_; ++i) {
      int state = all_dn_states[i];
      state_[state] = 1;
      spin_id_[state] = i;
      dn_states_[i] = state;
    }
    int j = 0;
    for (int i=num_dnspins_; i<num_sites_; ++i) {
      dnhole_states_[j++] = all_dn_states[i];
    }
  }
  else {
    int total_spins = num_upspins_+num_dnspins_;
    // DN spins
    int j = 0;
    for (int i=num_upspins_; i<total_spins; ++i) {
      int state = num_sites_+all_up_states[i];
      state_[state] = 1;
      spin_id_[state] = j;
      dn_states_[j++] = state;
    }
    // DN holes
    j = 0;
    for (int i=0; i<num_upspins_; ++i) {
      int state = num_sites_+all_up_states[i];
      dnhole_states_[j++] = state;
    }
    for (int i=total_spins; i<num_sites_; ++i) {
      int state = num_sites_+all_up_states[i];
      dnhole_states_[j++] = state;
    }
  }
  // number of doublely occupied sites
  num_dblocc_sites_ = 0;
  if (double_occupancy_) {
    for (int i=0; i<num_sites_; ++i) {
      if (state_[i]==1 && state_[i+num_sites_]==1)
        num_dblocc_sites_++;
    }
  }

  // Order the numbering of spins according to site index
  int i = 0;
  for (int s=0; s<num_sites_; ++s) {
    if (state_[s]) {
      spin_id_[i] = i;
      up_states_[i] = s;
      i++;
    }
  }
  i = 0;
  for (int s=num_sites_; s<num_states_; ++s) {
    if (state_[s]) {
      spin_id_[i] = i;
      dn_states_[i] = s;
      i++;
    }
  }
  /*
  std::cout << "Initial config = ";
  std::cout << *this;
  std::cout << "Up sites = ";
  for (const auto& s : upspin_sites()) std::cout<<s<<"| ";
  std::cout << "\nDp sites = ";
  for (const auto& s : dnspin_sites()) std::cout<<s<<"| ";
  std::cout << "\n";
  getchar();
  */
}

void FockBasis::set_custom(void)
{
  proposed_move_ = move_t::null;
  //ssign_ = 1;
  state_.setZero();
  std::vector<int> all_up_states(num_sites_);
  for (int i=0; i<num_sites_; ++i) all_up_states[i] = i;
  //std::shuffle(all_up_states.begin(),all_up_states.end(),rng_);
  for (int i=0; i<num_upspins_; ++i) {
    int state = all_up_states[i];
    state_[state] = 1;
    spin_id_[state] = i;
    up_states_[i] = state;
  }
  int j=0;
  for (int i=num_upspins_; i<num_sites_; ++i) {
    uphole_states_[j++] = all_up_states[i];
  }

  // DN spins & holes
  std::vector<int> all_dn_states(num_sites_);
  for (int i=0; i<num_sites_; ++i) all_dn_states[i] = num_sites_+i;
  //std::shuffle(all_dn_states.begin(),all_dn_states.end(),rng_);
  int last_site = num_sites_-1;
  for (int i=0; i<num_dnspins_; ++i) {
    int state = all_dn_states[last_site-i];
    state_[state] = 1;
    spin_id_[state] = i;
    dn_states_[i] = state;
  }
  j = 0;
  for (int i=num_dnspins_; i<num_sites_; ++i) {
    dnhole_states_[j++] = all_dn_states[last_site-i];
  }

  // number of doublely occupied sites
  num_dblocc_sites_ = 0;
  if (double_occupancy_) {
    for (int i=0; i<num_sites_; ++i) {
      if (state_[i]==1 && state_[i+num_sites_]==1)
        num_dblocc_sites_++;
    }
  }
}

bool FockBasis::gen_upspin_hop(void)
{
  if (proposed_move_!=move_t::null) undo_last_move();
  if (num_upholes_==0 || num_upspins_==0) {
    proposed_move_ = move_t::null;
    return false;
  }
  mv_upspin_ = rng_.random_upspin();
  mv_uphole_ = rng_.random_uphole();
  //std::cout << " rng test = " << spin_site_pair.first << "\n";
  up_fr_state_ = up_states_[mv_upspin_]; 
  up_to_state_ = uphole_states_[mv_uphole_]; 
  if (!double_occupancy_ && state_[num_sites_+up_to_state_]) {
    proposed_move_ = move_t::null;
    return false;
  }
  else {
    proposed_move_=move_t::upspin_hop;
    state_[up_fr_state_] = 0;
    state_[up_to_state_] = 1;
    nd_incr_frsite_ = -state_[num_sites_+up_fr_state_];
    nd_incr_tosite_ = +state_[num_sites_+up_to_state_]; 
    //dblocc_increament_ = state_[num_sites_+up_to_state_]; // must be 0 or 1
    //dblocc_increament_ -= state_[num_sites_+up_fr_state_];
    new_elems_.resize(2);
    new_elems_[0] = up_fr_state_;
    new_elems_[1] = up_to_state_;
    //fr_state = up_fr_state_;
    //to_state = up_to_state_;

    // sign (considered that the state is aready changed above)
    op_sign_ = 1;
    for (int i=up_to_state_+1; i<up_fr_state_; ++i) {
      if (state_[i]) op_sign_ = -op_sign_;
    }
    for (int i=up_fr_state_+1; i<up_to_state_; ++i) {
      if (state_[i]) op_sign_ = -op_sign_;
    }
    //ssign_ *= op_sign_;
    return true;
  }
}

bool FockBasis::gen_dnspin_hop(void)
{
  if (proposed_move_!=move_t::null) undo_last_move();
  if (num_dnholes_==0 || num_dnspins_==0) {
    proposed_move_ = move_t::null;
    return false;
  }
  mv_dnspin_ = rng_.random_dnspin();
  mv_dnhole_ = rng_.random_dnhole();
  //std::cout << " rng test = " << spin_site_pair.first << "\n";
  dn_fr_state_ = dn_states_[mv_dnspin_]; 
  dn_to_state_ = dnhole_states_[mv_dnhole_]; 
  dn_fr_site_ = dn_fr_state_-num_sites_;
  dn_to_site_ = dn_to_state_-num_sites_;
  if (!double_occupancy_ && state_[dn_to_site_]) {
    proposed_move_ = move_t::null;
    return false;
  }
  else {
    proposed_move_=move_t::dnspin_hop;
    state_[dn_fr_state_] = 0;
    state_[dn_to_state_] = 1;
    nd_incr_frsite_ = -state_[dn_fr_site_];
    nd_incr_tosite_ = +state_[dn_to_site_]; 
    //dblocc_increament_ = state_[dn_to_site_]; // must be 0 or 1
    //dblocc_increament_ -= state_[dn_fr_site_];
    new_elems_.resize(2);
    new_elems_[0] = dn_fr_state_;
    new_elems_[1] = dn_to_state_;
    // sign (considered that the state is aready changed above)
    op_sign_ = 1;
    for (int i=dn_to_state_+1; i<dn_fr_state_; ++i) {
      if (state_[i]) op_sign_ = -op_sign_;
    }
    for (int i=dn_fr_state_+1; i<dn_to_state_; ++i) {
      if (state_[i]) op_sign_ = -op_sign_;
    }
    //ssign_ *= op_sign_;
    return true;
  }
}

const int& FockBasis::which_upspin(void) const
{
  if (proposed_move_==move_t::upspin_hop) {
    return mv_upspin_;
  }
  else if (proposed_move_==move_t::exchange) {
    return mv_upspin_;
  }
  else {
    throw std::logic_error("FockBasis::which_upspin: no upspin move exists");
  }
}

const int& FockBasis::which_dnspin(void) const
{
  if (proposed_move_==move_t::dnspin_hop) {
    return mv_dnspin_;
  }
  else if (proposed_move_==move_t::exchange) {
    return mv_dnspin_;
  }
  else {
    throw std::logic_error("FockBasis::which_dnspin: no dnspin move exists");
  }
}

const int& FockBasis::which_frsite(void) const
{
  if (proposed_move_==move_t::upspin_hop) {
    return up_fr_state_;
  }
  else if (proposed_move_==move_t::dnspin_hop) {
    return dn_fr_site_;
  }
  else {
    throw std::logic_error("BasisState::which_frsite: no existing move");
  }
}

const int& FockBasis::which_site(void) const
{
  if (proposed_move_==move_t::upspin_hop) {
    return up_to_state_;
  }
  else if (proposed_move_==move_t::dnspin_hop) {
    return dn_to_site_;
  }
  else {
    throw std::logic_error("FockBasis::which_site: no existing move");
  }
}

const int& FockBasis::which_upspin_site(void) const
{
  if (proposed_move_==move_t::upspin_hop) {
    return up_to_state_;
  }
  else if (proposed_move_==move_t::exchange) {
    return up_to_state_;
  }
  else {
    throw std::logic_error("FockBasis::which_site: no existing move");
  }
}

const int& FockBasis::which_dnspin_site(void) const
{
  if (proposed_move_==move_t::dnspin_hop) {
    return dn_to_site_;
  }
  else if (proposed_move_==move_t::exchange) {
    return up_fr_state_; // since exchanging with 'up'-spin site
  }
  else {
    throw std::logic_error("FockBasis::which_site: no existing move");
  }
}


bool FockBasis::gen_exchange_move(void)
{
  if (proposed_move_!=move_t::null) undo_last_move();
  if (num_upholes_==0 || num_upspins_==0) return false;
  if (num_dnholes_==0 || num_dnspins_==0) return false;
  mv_upspin_ = rng_.random_upspin();
  mv_dnspin_ = rng_.random_dnspin();
  up_fr_state_ = up_states_[mv_upspin_]; 
  dn_fr_state_ = dn_states_[mv_dnspin_]; 
  up_to_state_ = dn_fr_state_-num_sites_; 
  dn_to_state_ = up_fr_state_+num_sites_; 
  mv_uphole_ = -1;
  for (int i=0; i<num_upholes_; ++i) {
    if (uphole_states_[i]==up_to_state_) {
      mv_uphole_ = i;
      break;
    }
  }
  if (mv_uphole_<0) return false;
  mv_dnhole_ = -1;
  for (int i=0; i<num_dnholes_; ++i) {
    if (dnhole_states_[i]==dn_to_state_) {
      mv_dnhole_ = i;
      break;
    }
  }
  if (mv_dnhole_<0) return false;
  // valid move
  proposed_move_ = move_t::exchange;
  state_[up_fr_state_] = 0;
  state_[up_to_state_] = 1;
  state_[dn_fr_state_] = 0;
  state_[dn_to_state_] = 1;
  new_elems_.resize(4);
  new_elems_[0] = up_fr_state_;
  new_elems_[1] = up_to_state_;
  new_elems_[2] = dn_fr_state_;
  new_elems_[3] = dn_to_state_;
  // sign (considered that the state is aready changed above)
  op_sign_ = 1;
  for (int i=up_to_state_+1; i<up_fr_state_; ++i) {
    if (state_[i]) op_sign_ = -op_sign_;
  }
  for (int i=up_fr_state_+1; i<up_to_state_; ++i) {
    if (state_[i]) op_sign_ = -op_sign_;
  }
  for (int i=dn_to_state_+1; i<dn_fr_state_; ++i) {
    if (state_[i]) op_sign_ = -op_sign_;
  }
  for (int i=dn_fr_state_+1; i<dn_to_state_; ++i) {
    if (state_[i]) op_sign_ = -op_sign_;
  }
  //ssign_ *= op_sign_;
  return true;
}

int FockBasis::op_ni_up(const int& site) const
{
  return state_[site];
}

int FockBasis::op_ni_dn(const int& site) const
{
  return state_[num_sites_+site];
}

int FockBasis::op_ni_updn(const int& site) const
{
  if (state_[site] && state_[num_sites_+site]) return 1;
  else return 0;
}

int FockBasis::op_ni_dblon(const int& site) const
{
  if (state_[site] && state_[num_sites_+site]) return 1;
  else return 0;
}

int FockBasis::op_ni_holon(const int& site) const
{
  if (state_[site] || state_[num_sites_+site]) return 0;
  else return 1;
}

int FockBasis::op_Sz(const int& site) const
{
  return state_[site]-state_[num_sites_+site];
}

bool FockBasis::op_cdagc_up(const int& fr_site, const int& to_site) const
{
  if (proposed_move_!=move_t::null) undo_last_move();
  if (state_[fr_site]==1 && state_[to_site]==0) {
    up_fr_state_ = fr_site;
    up_to_state_ = to_site;
  }
  else return false;
  mv_upspin_ = spin_id_[up_fr_state_];
  op_sign_ = 1;
  if (up_fr_state_==up_to_state_ && state_[up_fr_state_]) {
    nd_incr_frsite_ = 0;
    nd_incr_tosite_ = 0;
    //dblocc_increament_ = 0;
    return true;
  }
  // actual move now
  proposed_move_ = move_t::upspin_hop;
  state_[up_fr_state_] = 0;
  state_[up_to_state_] = 1;
  // change in no of doubly occupied sites
  nd_incr_frsite_ = -state_[num_sites_+up_fr_state_];
  nd_incr_tosite_ = +state_[num_sites_+up_to_state_]; 
  //dblocc_increament_ = state_[num_sites_+up_to_state_]; // must be 0 or 1
  //dblocc_increament_ -= state_[num_sites_+up_fr_state_];
  // sign (considered that the state is aready changed above)
  for (int i=up_to_state_+1; i<up_fr_state_; ++i) {
    if (state_[i]) op_sign_ = -op_sign_;
  }
  for (int i=up_fr_state_+1; i<up_to_state_; ++i) {
    if (state_[i]) op_sign_ = -op_sign_;
  }
  /*
  if (op_sign_==-1) {
    std::cout << "up_fr_state = " << up_fr_state_ << "\n";
    std::cout << "up_to_state = " << up_to_state_ << "\n";
    std::cout << "state: |" << state_.transpose() << ">\n";
    std::cout << "sign(op_cdagc_up) = "<<op_sign_<<"\n"; getchar();
  }
  */

  new_elems_.resize(2);
  new_elems_[0] = up_fr_state_;
  new_elems_[1] = up_to_state_;
  return true;
}

bool FockBasis::op_cdagc2_up(const int& site_i, const int& site_j) const
{
  if (proposed_move_!=move_t::null) undo_last_move();
  if (state_[site_i]==0 && state_[site_j]==1) {
    up_fr_state_ = site_j;
    up_to_state_ = site_i;
  }
  else if (state_[site_i]==1 && state_[site_j]==0) {
    up_fr_state_ = site_i;
    up_to_state_ = site_j;
  }
  else return false;
  mv_upspin_ = spin_id_[up_fr_state_];
  op_sign_ = 1;
  if (up_fr_state_==up_to_state_ && state_[up_fr_state_]) {
    nd_incr_frsite_ = 0;
    nd_incr_tosite_ = 0;
    //dblocc_increament_ = 0;
    return true;
  }

  // actual move now
  proposed_move_ = move_t::upspin_hop;
  state_[up_fr_state_] = 0;
  state_[up_to_state_] = 1;
  // change in no of doubly occupied sites
  nd_incr_frsite_ = -state_[num_sites_+up_fr_state_];
  nd_incr_tosite_ = +state_[num_sites_+up_to_state_]; 
  //dblocc_increament_ = state_[num_sites_+up_to_state_]; // must be 0 or 1
  //dblocc_increament_ -= state_[num_sites_+up_fr_state_];
  // sign (considered that the state is aready changed above)
  for (int i=up_to_state_+1; i<up_fr_state_; ++i) {
    if (state_[i]) op_sign_ = -op_sign_;
  }
  for (int i=up_fr_state_+1; i<up_to_state_; ++i) {
    if (state_[i]) op_sign_ = -op_sign_;
  }
  /*
  if (op_sign_==-1) {
    std::cout << "up_fr_state = " << up_fr_state_ << "\n";
    std::cout << "up_to_state = " << up_to_state_ << "\n";
    std::cout << "state: |" << state_.transpose() << ">\n";
    std::cout << "sign(op_cdagc_up) = "<<op_sign_<<"\n"; getchar();
  }
  */

  new_elems_.resize(2);
  new_elems_[0] = up_fr_state_;
  new_elems_[1] = up_to_state_;
  return true;
}

bool FockBasis::op_cdagc_dn(const int& fr_site, const int& to_site) const
{
  if (proposed_move_!=move_t::null) undo_last_move();
  int idx_i = num_sites_+fr_site;
  int idx_j = num_sites_+to_site;
  if (state_[idx_i]==1 && state_[idx_j]==0) {
    dn_fr_state_ = idx_i;
    dn_to_state_ = idx_j;
  }
  else return false;
  mv_dnspin_ = spin_id_[dn_fr_state_];
  dn_fr_site_ = dn_fr_state_-num_sites_;
  dn_to_site_ = dn_to_state_-num_sites_;
  op_sign_ = 1;
  if (dn_fr_state_==dn_to_state_ && state_[dn_fr_state_]) {
    nd_incr_frsite_ = 0;
    nd_incr_tosite_ = 0;
    //dblocc_increament_ = 0;
    return true;
  }
  // actual move now
  proposed_move_ = move_t::dnspin_hop;
  state_[dn_fr_state_] = 0;
  state_[dn_to_state_] = 1;
  // change in no of doubly occupied sites
  nd_incr_frsite_ = -state_[dn_fr_site_];
  nd_incr_tosite_ = +state_[dn_to_site_]; 
  //dblocc_increament_ = state_[dn_to_site_]; // must be 0 or 1
  //dblocc_increament_ -= state_[dn_fr_site_];
  // sign (considered that the state is aready changed above)
  for (int i=dn_to_state_+1; i<dn_fr_state_; ++i) {
    if (state_[i]) op_sign_ = -op_sign_;
  }
  for (int i=dn_fr_state_+1; i<dn_to_state_; ++i) {
    if (state_[i]) op_sign_ = -op_sign_;
  }

  /*
  if (op_sign_==-1) {
    std::cout << "dn_fr_state = " << dn_fr_state_ << "\n";
    std::cout << "dn_to_state = " << dn_to_state_ << "\n";
    std::cout << "state: |" << state_.transpose() << ">\n";
    std::cout << "sign(op_cdagc_up) = "<<op_sign_<<"\n"; getchar();
  }
  */

  new_elems_.resize(2);
  new_elems_[0] = dn_fr_state_;
  new_elems_[1] = dn_to_state_;
  return true;
}


bool FockBasis::op_cdagc2_dn(const int& site_i, const int& site_j) const
{
  if (proposed_move_!=move_t::null) undo_last_move();
  int idx_i = num_sites_+site_i;
  int idx_j = num_sites_+site_j;
  if (state_[idx_i]==0 && state_[idx_j]==1) {
    dn_fr_state_ = idx_j; 
    dn_to_state_ = idx_i; 
  }
  else if (state_[idx_i]==1 && state_[idx_j]==0) {
    dn_fr_state_ = idx_i;
    dn_to_state_ = idx_j;
  }
  else return false;
  mv_dnspin_ = spin_id_[dn_fr_state_];
  dn_fr_site_ = dn_fr_state_-num_sites_;
  dn_to_site_ = dn_to_state_-num_sites_;
  op_sign_ = 1;
  if (dn_fr_state_==dn_to_state_ && state_[dn_fr_state_]) {
    nd_incr_frsite_ = 0;
    nd_incr_tosite_ = 0;
    //dblocc_increament_ = 0;
    return true;
  }
  // actual move now
  proposed_move_ = move_t::dnspin_hop;
  state_[dn_fr_state_] = 0;
  state_[dn_to_state_] = 1;
  // change in no of doubly occupied sites
  nd_incr_frsite_ = -state_[dn_fr_site_];
  nd_incr_tosite_ = +state_[dn_to_site_]; 
  //dblocc_increament_ = state_[dn_to_site_]; // must be 0 or 1
  //dblocc_increament_ -= state_[dn_fr_site_];
  // sign (considered that the state is aready changed above)
  for (int i=dn_to_state_+1; i<dn_fr_state_; ++i) {
    if (state_[i]) op_sign_ = -op_sign_;
  }
  for (int i=dn_fr_state_+1; i<dn_to_state_; ++i) {
    if (state_[i]) op_sign_ = -op_sign_;
  }

  /*
  if (op_sign_==-1) {
    std::cout << "dn_fr_state = " << dn_fr_state_ << "\n";
    std::cout << "dn_to_state = " << dn_to_state_ << "\n";
    std::cout << "state: |" << state_.transpose() << ">\n";
    std::cout << "sign(op_cdagc_up) = "<<op_sign_<<"\n"; getchar();
  }
  */

  new_elems_.resize(2);
  new_elems_[0] = dn_fr_state_;
  new_elems_[1] = dn_to_state_;
  return true;
}


int FockBasis::op_exchange_ud(const int& site_i, const int& site_j) const
{
  if (proposed_move_!=move_t::null) undo_last_move();
  if (site_i == site_j) return 1;
  auto* ni_up = &state_[site_i];
  auto* nj_up = &state_[site_j];
  auto* ni_dn = &state_[num_sites_+site_i];
  auto* nj_dn = &state_[num_sites_+site_j];
  if (*ni_up==1 && *nj_up==0 && *ni_dn==0 && *nj_dn==1) {
    *ni_up = 0;
    *nj_up = 1;
    *ni_dn = 1;
    *nj_dn = 0;
    up_fr_state_ = site_i;
    up_to_state_ = site_j;
    dn_fr_state_ = num_sites_+site_j;
    dn_to_state_ = num_sites_+site_i;
  }
  else if (*ni_up==0 && *nj_up==1 && *ni_dn==1 && *nj_dn==0) {
    *ni_up = 1;
    *nj_up = 0;
    *ni_dn = 0;
    *nj_dn = 1;
    up_fr_state_ = site_j;
    up_to_state_ = site_i;
    dn_fr_state_ = num_sites_+site_i;
    dn_to_state_ = num_sites_+site_j;
  }
  else {
    return 0;
  }
  // sign (considered that the state is aready changed above)
  op_sign_ = 1;
  for (int i=up_to_state_+1; i<up_fr_state_; ++i) {
    if (state_[i]) op_sign_ = -op_sign_;
  }
  for (int i=up_fr_state_+1; i<up_to_state_; ++i) {
    if (state_[i]) op_sign_ = -op_sign_;
  }
  for (int i=dn_to_state_+1; i<dn_fr_state_; ++i) {
    if (state_[i]) op_sign_ = -op_sign_;
  }
  for (int i=dn_fr_state_+1; i<dn_to_state_; ++i) {
    if (state_[i]) op_sign_ = -op_sign_;
  }
  proposed_move_ = move_t::exchange;
  return op_sign_;
}

void FockBasis::commit_last_move(void)
{
  // double occupancy count
  switch (proposed_move_) {
    case move_t::upspin_hop:
      //num_dblocc_sites_ += dblocc_increament_;
      //operator[](up_fr_state_) = 0;
      //operator[](up_to_state_) = 1;
      spin_id_[up_fr_state_] = null_id_;
      spin_id_[up_to_state_] = mv_upspin_;
      up_states_[mv_upspin_] = up_to_state_;
      uphole_states_[mv_uphole_] = up_fr_state_;
      proposed_move_ = move_t::null;
      break;
    case move_t::dnspin_hop:
      //num_dblocc_sites_ += dblocc_increament_;
      //operator[](dn_fr_state_) = 0;
      //operator[](dn_to_state_) = 1;
      spin_id_[dn_fr_state_] = null_id_;
      spin_id_[dn_to_state_] = mv_dnspin_;
      dn_states_[mv_dnspin_] = dn_to_state_;
      dnhole_states_[mv_dnhole_] = dn_fr_state_;
      proposed_move_ = move_t::null;
      break;
    case move_t::exchange:
      spin_id_[up_fr_state_] = null_id_;
      spin_id_[up_to_state_] = mv_upspin_;
      spin_id_[dn_fr_state_] = null_id_;
      spin_id_[dn_to_state_] = mv_dnspin_;
      up_states_[mv_upspin_] = up_to_state_;
      uphole_states_[mv_uphole_] = up_fr_state_;
      dn_states_[mv_dnspin_] = dn_to_state_;
      dnhole_states_[mv_dnhole_] = dn_fr_state_;
      proposed_move_ = move_t::null;
      break;
    case move_t::null:
      break;
  }
  // check
  /*
  int m = 0;
  int n = 0;
  for (int i=0; i<num_sites_; ++i) m += operator[](i);
  for (int i=num_sites_; i<num_states_; ++i) n += operator[](i);
  if (m!= num_upspins_ || n!= num_dnspins_) {
    throw std::logic_error("FockBasis::commit_last_move");
  }*/
}


void FockBasis::undo_last_move(void) const
{
  // double occupancy count
  switch (proposed_move_) {
    case move_t::upspin_hop:
      state_[up_fr_state_] = 1;
      state_[up_to_state_] = 0;
      break;
    case move_t::dnspin_hop:
      state_[dn_fr_state_] = 1;
      state_[dn_to_state_] = 0;
      break;
    case move_t::exchange:
      state_[up_fr_state_] = 1;
      state_[up_to_state_] = 0;
      state_[dn_fr_state_] = 1;
      state_[dn_to_state_] = 0;
      break;
    case move_t::null:
      break;
  }
  //dblocc_increament_ = 0;
  nd_incr_frsite_ = 0;
  nd_incr_tosite_ = 0;
  proposed_move_ = move_t::null;
}

std::ostream& operator<<(std::ostream& os, const FockBasis& bs)
{
  os << "state: |";
  for (int i=0; i<bs.num_sites_; ++i) os << bs.state_[i] << " ";
  os << ": ";
  for (int i=bs.num_sites_; i<bs.num_states_; ++i) os << bs.state_[i] << " ";
  os << ">";
  //os << "state: |" << bs.state_.transpose() << ">\n";
  return os;
}


} // end namespace vmc
