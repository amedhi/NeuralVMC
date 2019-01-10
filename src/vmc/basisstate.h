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
#include <utility>
#include <bitset>
#include <array>
#include "./random.h"

namespace vmc {

enum class spin {UP, DN};

enum class move_t {upspin_hop, dnspin_hop, exchange, null};

class SiteState : public std::bitset<2>
{
public:
  SiteState()
    : null_id_{100000}  
    , id_{null_id_,null_id_}
    , spin_UP_{static_cast<size_t>(spin::UP)}
    , spin_DN_{static_cast<size_t>(spin::DN)}
  { 
    reset(); 
  }
  void put_upspin(const unsigned& n) {set(spin_UP_); id_.first=n;} 
  void put_uphole(const unsigned& n) {reset(spin_UP_); id_.first=n;} 
  void put_dnspin(const unsigned& n) {set(spin_DN_); id_.second=n;} 
  void put_dnhole(const unsigned& n) {reset(spin_DN_); id_.second=n;} 
  const unsigned& upspin_id(void) const { return id_.first; }
  const unsigned& dnspin_id(void) const { return id_.second; }
  const unsigned& uphole_id(void) const { return id_.first; }
  const unsigned& dnhole_id(void) const { return id_.second; }
  int num_upspins(void) const { return operator[](spin_UP_); }  
  int num_dnspins(void) const { return operator[](spin_DN_); }  
  bool have_upspin(void) const { return test(spin_UP_); }  
  bool have_dnspin(void) const { return test(spin_DN_); }  
  bool have_uphole(void) const { return !test(spin_UP_); }  
  bool have_dnhole(void) const { return !test(spin_DN_); }  
  bool is_full(void) const { return all(); }  
  bool is_empty(void) const { return none(); }  
  bool is_not_empty(void) const { return any(); }  
  int occupancy(void) const { return count(); }  
private:
  using size_t = std::size_t;
  unsigned null_id_{100000};
  std::pair<unsigned,unsigned> id_;
  unsigned spin_UP_{0};
  unsigned spin_DN_{1};
};

class BasisState : public std::vector<SiteState> 
{
public:
  BasisState(); 
  BasisState(const unsigned& num_sites); 
  BasisState(const unsigned& num_sites, const bool& allow_dbl);
  ~BasisState() {} 
  void set_vaccuum(const unsigned& num_sites, const bool& allow_dbl=true);
  void allow_double_occupancy(const bool& allow);
  void init_spins(const unsigned& num_upspins, const unsigned& num_dnspins);
  void set_random(void);
  const bool& double_occupancy(void) const { return double_occupancy_; }
  std::pair<int,int> gen_upspin_hop(void);
  std::pair<int,int> gen_dnspin_hop(void);
  std::pair<int,int> gen_exchange_move(void);
  std::pair<int,int> exchange_move_part(void);
  const int& dblocc_count(void) const { return num_dblocc_sites_; }
  const int& dblocc_increament(void) const { return dblocc_increament_; }
  const unsigned& site_capacity(void) const { return site_capacity_; }
  void accept_last_move(void);
  const std::vector<int>& upspin_sites(void) const { return upspin_sites_; }
  const std::vector<int>& dnspin_sites(void) const { return dnspin_sites_; }
  RandomGenerator& rng(void) const { return rng_; }
  //const SiteState& upspin(unsigned)
  //bool annihilate(i,sigma);
  //bool hop(i,j,sigma);
  friend std::ostream& operator<<(std::ostream& os, const BasisState& bs);
private:
  mutable RandomGenerator rng_;
  unsigned num_sites_{0};
  unsigned num_upspins_{0};
  unsigned num_dnspins_{0};
  unsigned num_upholes_{0};
  unsigned num_dnholes_{0};
  int num_dblocc_sites_{0};
  bool double_occupancy_{true};
  unsigned site_capacity_{2};
  std::vector<int> upspin_sites_;
  std::vector<int> dnspin_sites_;
  std::vector<int> uphole_sites_;
  std::vector<int> dnhole_sites_;

  // update moves
  move_t proposed_move_;
  int dblocc_increament_{0};
  //move_type accepted_move;
  int mv_upspin_;
  int mv_uphole_;
  int up_tosite_;
  int mv_dnspin_;
  int mv_dnhole_;
  int dn_tosite_;

  void clear(void); 
};


/*
class SpinParticle 
{
public:
  SpinParticle() : id_{0}, site_{0}, sigma_{spin::HL} {}
  SpinParticle(const spin& s) : id_{0}, site_{0}, sigma_{s} {}
  SpinParticle(const unsigned& i, const spin& s) : id_{0}, site_{i}, sigma_{s} {}
  const spin& sigma(void) const { return sigma_; }
  const unsigned& id(void) const { return id_; }
protected:
  unsigned id_;
private:
  unsigned site_;
  spin sigma_;
};

class SpinUp : public SpinParticle 
{
public:
  SpinUp() {}
  SpinUp(const unsigned& site) : SpinParticle{site, spin::UP} { id_=total_num_++; }
  SpinUp(const SpinUp& p) : SpinParticle(p) { total_num_++; }
  ~SpinUp() { --total_num_; }
private:
  static unsigned total_num_;
};

class SpinDn : public SpinParticle 
{
public:
  SpinDn(const unsigned& site) : SpinParticle{site, spin::DN} { id_=total_num_++; }
  SpinDn(const SpinDn& p) : SpinParticle(p) { total_num_++; }
  ~SpinDn() { --total_num_; }
  static unsigned total_num_;
private:
};

class Hole : public SpinParticle 
{
public:
  Hole(const unsigned& site) : SpinParticle{site, spin::HL} { id_=total_num_++; }
  Hole(const Hole& p) : SpinParticle(p) { total_num_++; }
  ~Hole() { --total_num_; }
  static unsigned num_holes(void) { return total_num_; }
private:
  static unsigned total_num_;
};

class Doublon : public SpinParticle 
{
public:
  Doublon() : SpinParticle(spin::UD) { id_=total_num_++; }
  ~Doublon() { --total_num_; }
private:
  unsigned up_id_;
  unsigned dn_id_;
  static unsigned total_num_;
};
using site_state = SpinParticle*;

*/




} // end namespace vmc

#endif