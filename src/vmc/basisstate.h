/*---------------------------------------------------------------------------
* Author: Amal Medhi
* Date:   2017-02-13 10:16:02
* Last Modified by:   Amal Medhi, amedhi@macbook
* Last Modified time: 2017-04-12 21:26:24
* Copyright (C) Amal Medhi, amedhi@iisertvm.ac.in
*----------------------------------------------------------------------------*/
#ifndef BASISSTATE_H
#define BASISSTATE_H

#include <iostream>
#include <vector>
#include <Eigen/Core>
#include "./random.h"
#include "../wavefunction/matrix.h"

namespace vmc {

enum class move_t {upspin_hop, dnspin_hop, exchange, null};

class FockBasis 
{
public:
  FockBasis() { init(0); }
  FockBasis(const int& num_sites, const bool& allow_dbl=true) 
    { init(num_sites, allow_dbl); }
  ~FockBasis() {}
  RandomGenerator& rng(void) const { return rng_; }
  void init(const int& num_sites, const bool& allow_dbl=true);
  void init_spins(const int& num_upspins, const int& num_dnspins);
  const eig::ivector& state(void) const { return state_; }
  const std::vector<int>& upspin_sites(void) const { return up_states_; }
  const std::vector<int>& dnspin_sites(void) const 
  { 
    for (int i=0; i<num_dnspins_; ++i) dnspin_sites_[i] = dn_states_[i]-num_sites_;
    return dnspin_sites_; 
  }
  void set_random(void);
  void set_custom(void);
  bool double_occupancy(void) const { return double_occupancy_; }
  bool gen_upspin_hop(void);
  bool gen_dnspin_hop(void);
  bool gen_exchange_move(void);
  const int& which_upspin(void) const;
  const int& which_dnspin(void) const;
  const int& which_site(void) const; 
  const int& which_upspin_site(void) const; 
  const int& which_dnspin_site(void) const; 
  const std::vector<int> new_elems(void) const { return new_elems_; }
  void commit_last_move(void);
  void undo_last_move(void) const;
  int op_ni_up(const int& site) const;
  int op_ni_dn(const int& site) const;
  int op_ni_updn(const int& site) const;
  bool op_cdagc_up(const int& fr_site, const int& to_site) const;
  bool op_cdagc_dn(const int& fr_site, const int& to_site) const;
  int op_exchange_ud(const int& fr_site, const int& to_site) const;
  //const int state_sign(void) const { return ssign_; }
  const int op_sign(void) const { return op_sign_; }
  const int delta_nd(void) const { return dblocc_increament_; }
  friend std::ostream& operator<<(std::ostream& os, const FockBasis& bs);
private:
  mutable RandomGenerator rng_;
  mutable eig::ivector state_;
  //mutable int ssign_; // sign of the state
  eig::ivector spin_id_;  // store which UP-spin for a given state index
  int num_sites_{0};
  int num_states_{0};
  int num_upspins_{0};
  int num_dnspins_{0};
  int num_upholes_{0};
  int num_dnholes_{0};
  int num_dblocc_sites_{0};
  bool double_occupancy_{true};
  std::vector<int> up_states_;
  std::vector<int> dn_states_;
  std::vector<int> uphole_states_;
  std::vector<int> dnhole_states_;
  mutable std::vector<int> dnspin_sites_;

  // update moves
  mutable move_t proposed_move_;
  mutable int dblocc_increament_{0};
  //move_type accepted_move;
  mutable int mv_upspin_;
  mutable int mv_uphole_;
  mutable int up_fr_state_;
  mutable int up_to_state_;
  mutable int dn_fr_state_;
  mutable int dn_to_state_;
  mutable int dn_fr_site_;
  mutable int dn_to_site_;
  mutable int mv_dnspin_;
  mutable int mv_dnhole_;
  mutable int dn_tostate_;
  mutable int op_sign_;
  mutable std::vector<int> new_elems_;
  int null_id_{-1};
  void clear(void); 
};
 



#ifdef OLD
class FockBasis //: public eig::ivec
{
public:
  FockBasis() { init(0); }
  FockBasis(const int& num_sites, const bool& allow_dbl=true) 
    { init(num_sites, allow_dbl); }
  ~FockBasis() {}
  RandomGenerator& rng(void) const { return rng_; }
  void init(const int& num_sites, const bool& allow_dbl=true);
  void init_spins(const int& num_upspins, const int& num_dnspins);
  const eig::ivec& state(void) const { return basis_; }
  void set_random(void);
  bool double_occupancy(void) const { return double_occupancy_; }
  bool gen_upspin_hop(void);
  bool gen_dnspin_hop(void);
  bool gen_exchange_move(void);
  void commit_last_move(void);
  void undo_last_move(void) const;
  int op_ni_up(const int& site) const;
  int op_ni_dn(const int& site) const;
  int op_ni_updn(const int& site) const;
  bool op_cdagc_up(const int& fr_site, const int& to_site) const;
  bool op_cdagc_dn(const int& fr_site, const int& to_site) const;
  int op_exchange_ud(const int& fr_site, const int& to_site) const;
  const int op_sign(void) const { return op_sign_; }
  const int delta_nd(void) const { return dblocc_increament_; }
private:
  mutable RandomGenerator rng_;
  mutable eig::ivec basis_;
  int num_sites_{0};
  int num_states_{0};
  int num_upspins_{0};
  int num_dnspins_{0};
  int num_upholes_{0};
  int num_dnholes_{0};
  int num_dblocc_sites_{0};
  bool double_occupancy_{true};
  std::vector<int> up_states_;
  std::vector<int> dn_states_;
  std::vector<int> uphole_states_;
  std::vector<int> dnhole_states_;

  // update moves
  mutable move_t proposed_move_;
  mutable int dblocc_increament_{0};
  //move_type accepted_move;
  mutable int mv_upspin_;
  mutable int mv_uphole_;
  mutable int up_fr_state_;
  mutable int up_to_state_;
  mutable int dn_fr_state_;
  mutable int dn_to_state_;
  mutable int up_tostate_;
  mutable int mv_dnspin_;
  mutable int mv_dnhole_;
  mutable int dn_tostate_;
  mutable int op_sign_;
  void clear(void); 
};
#endif 


} // end namespace vmc

#endif