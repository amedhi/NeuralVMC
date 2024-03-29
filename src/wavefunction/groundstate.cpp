/*---------------------------------------------------------------------------
* Author: Amal Medhi
* Date:   2017-03-20 09:43:12
* Last Modified by:   Amal Medhi, amedhi@macbook
* Last Modified time: 2017-04-13 15:03:31
* Copyright (C) Amal Medhi, amedhi@iisertvm.ac.in
*----------------------------------------------------------------------------*/
#include <stdexcept>
#include <boost/algorithm/string.hpp>
#include "./groundstate.h"

namespace var {

void GroundState::update(const input::Parameters& inputs)
{
  throw std::runtime_error("GroundState::update_parameters: function must be overriden");
}

void GroundState::update(const var::parm_vector& pvector, const unsigned& start_pos)
{
  throw std::runtime_error("GroundState::update_parameters: function must be overriden");
}

void GroundState::update(const lattice::Lattice& lattice)
{
  throw std::runtime_error("GroundState::update: function must be overriden");
}

void GroundState::get_wf_amplitudes(Matrix& psi)
{
  throw std::runtime_error("GroundState::get_wf_amplitudes: function must be overriden");
}

void GroundState::get_wf_amplitudes(Matrix& psiup, Matrix& psidn)
{
  throw std::runtime_error("GroundState::get_wf_amplitudes: function must be overriden");
}

void GroundState::get_wf_gradient(std::vector<Matrix>& psi_grad)
{
  throw std::runtime_error("GroundState::get_wf_gradient: function must be overriden");
}

void GroundState::get_wf_gradient(std::vector<Matrix>& psiup_grad, std::vector<Matrix>& psidn_grad) 
{
  throw std::runtime_error("GroundState::get_wf_gradient: function must be overriden");
}

std::string GroundState::info_str(void) const
{
  throw std::runtime_error("GroundState::info_str: function must be overriden");
}

void GroundState::set_particle_num(const input::Parameters& inputs)
{
  hole_doping_inp_ = inputs.set_value("hole_doping", 0.0);
  if (std::abs(last_hole_doping_-hole_doping_inp_)<1.0E-15) {
    // no change in hole doping, particle number remails same
    return;
  }
  last_hole_doping_ = hole_doping_inp_;
  band_filling_ = 1.0-hole_doping_inp_;

  std::string lname = inputs.set_value("lattice", "SQUARE");
  boost::to_upper(lname);
  if (lname=="NICKELATE" || lname=="NICKELATE_2D" || lname=="NICKELATE_2L") {
    int num_unitcells = num_sites_/2;
    num_upspins_ = static_cast<int>(std::round(0.5*band_filling_*num_unitcells));
    num_dnspins_ = num_upspins_;
    num_spins_ = num_upspins_ + num_dnspins_;
    band_filling_ = static_cast<double>(num_spins_)/num_unitcells;
    hole_doping_ = 1.0 - band_filling_;
    //std::cout << "num_spins = " << num_spins_ << "\n"; 
    //std::cout << "hole_doping_ = " << hole_doping_ << "\n"; getchar();
    return;
  }


  band_filling_ = 1.0-hole_doping_inp_;
  int num_sites = static_cast<int>(num_sites_);
  if (nonmagnetic_) {
    int n = static_cast<int>(std::round(0.5*band_filling_*num_sites));
    if (n<0 || n>num_sites) throw std::range_error("GroundState::set_particle_num:: hole doping out-of-range");
    num_upspins_ = static_cast<unsigned>(n);
    num_dnspins_ = num_upspins_;
    num_spins_ = num_upspins_ + num_dnspins_;
    band_filling_ = static_cast<double>(2*n)/num_sites;
    /*
    std::cout << "num_sites = " << num_sites_ << "\n";
    std::cout << "num_upspins = " << num_upspins_ << "\n";
    std::cout << "num_dnspins = " << num_dnspins_ << "\n";
    std::cout << "band_filling = " << band_filling_ << "\n";
    getchar();
    */
  }
  else{
    int n = static_cast<int>(std::round(band_filling_*num_sites));
    if (n<0 || n>2*num_sites) throw std::range_error("GroundState::set_particle_num:: hole doping out-of-range");
    num_spins_ = static_cast<unsigned>(n);
    num_dnspins_ = num_spins_/2;
    num_upspins_ = num_spins_ - num_dnspins_;
    band_filling_ = static_cast<double>(n)/num_sites;
  }
  hole_doping_ = 1.0 - band_filling_;
  //std::cout << "num_upspins_ = " << num_upspins_ << "\n";
  //std::cout << "num_dnspins_ = " << num_dnspins_ << "\n";
}

void GroundState::reset_spin_num(const int& num_upspin, const int& num_dnspin)
{
  if ((num_upspin+num_dnspin) != num_spins_) {
    throw std::range_error("GroundState::reset_spin_num:: spin counts does not match");
  }
  num_upspins_ = num_upspin;
  num_dnspins_ = num_dnspin;
  nonmagnetic_ = (num_upspins_ == num_dnspins_);
}

double GroundState::get_noninteracting_mu(void)
{
  std::vector<double> ek;
  for (int k=0; k<num_kpoints_; ++k) {
    Vector3d kvec = blochbasis_.kvector(k);
    mf_model_.construct_kspace_block(kvec);
    // spin-up states
    es_k_up.compute(mf_model_.quadratic_spinup_block(), Eigen::EigenvaluesOnly);
    ek.insert(ek.end(),es_k_up.eigenvalues().data(),
      es_k_up.eigenvalues().data()+kblock_dim_);
    // spin-dn states
    es_k_up.compute(mf_model_.quadratic_spindn_block(), Eigen::EigenvaluesOnly);
    ek.insert(ek.end(),es_k_up.eigenvalues().data(),
      es_k_up.eigenvalues().data()+kblock_dim_);
  }
  // sort energy levels
  std::sort(ek.begin(),ek.end());
  double mu;
  if (num_spins_ < ek.size()) {
    mu = 0.5*(ek[num_spins_-1]+ek[num_spins_]);
  }
  else mu = ek[num_spins_-1];
  //std::cout << "mu_0 = " << mu << "\n";
  return mu;
}

void GroundState::set_ft_matrix(const lattice::Lattice& lattice)
{
  // matrix for transformation from site-basis to k-basis
  FTU_.resize(num_kpoints_,num_kpoints_);
  double one_by_sqrt_nk = 1.0/std::sqrt(static_cast<double>(num_kpoints_));
  unsigned i = 0;
  for (unsigned n=0; n<num_kpoints_; ++n) {
    auto Ri = lattice.site(i).cell_coord();
    //std::cout << Ri << "\n"; getchar();
    for (unsigned k=0; k<num_kpoints_; ++k) {
      Vector3d kvec = blochbasis_.kvector(k);
      FTU_(n,k) = std::exp(ii()*kvec.dot(Ri)) * one_by_sqrt_nk;
    }
    i += kblock_dim_;
    // i is first basis site in next unitcell
  }
  //std::cout << FTU_ << "\n"; 
  //getchar();
}


} // end namespace var
