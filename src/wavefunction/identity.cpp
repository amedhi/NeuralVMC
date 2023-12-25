/*---------------------------------------------------------------------------
* @Author: Amal Medhi, amedhi@mbpro
* @Date:   2019-02-07 12:31:24
* @Last Modified by:   Amal Medhi
* @Last Modified time: 2023-10-12 13:01:41
*----------------------------------------------------------------------------*/
#include "./identity.h"

namespace var {

Identity::Identity(const MF_Order::order_t& order, const input::Parameters& inputs, 
    const lattice::Lattice& lattice) 
  : GroundState(order, MF_Order::pairing_t::null)
{
  init(inputs, lattice);
}

int Identity::init(const input::Parameters& inputs, 
  const lattice::Lattice& lattice)
{
  name_ = "Identity";
  order_name_ = "Identity";
  // sites & bonds
  num_sites_ = lattice.num_sites();
  num_bonds_ = lattice.num_bonds();

  // bloch basis
  blochbasis_.construct(lattice);
  num_kpoints_ = blochbasis_.num_kpoints();
  kblock_dim_ = blochbasis_.subspace_dimension();
  // FT matrix for transformation from 'site basis' to k-basis
  set_ft_matrix(lattice);

  // particle number
  set_particle_num(inputs);
  varparms_.clear();
  num_varparms_ = 0;
  return 0;
}

std::string Identity::info_str(void) const
{
  std::ostringstream info;
  info << "# Ground State: '"<<name_<<" ("<<order_name_<<")'\n";
  info << "# Hole doping = "<<hole_doping()<<"\n";
  info << "# Particles = "<< num_upspins()+num_dnspins();
  info << " (Nup = "<<num_upspins()<<", Ndn="<<num_dnspins()<<")\n";
  return info.str();
}

void Identity::update(const input::Parameters& inputs)
{
  set_particle_num(inputs);
}

void Identity::update(const var::parm_vector& pvector, const unsigned& start_pos)
{
  return;
}

void Identity::get_wf_amplitudes(Matrix& psi) 
{
  psi.setOnes();
}

void Identity::get_wf_gradient(std::vector<Matrix>& psi_gradient) 
{
  for (auto& m : psi_gradient) m.setZero();
}



} // end namespace var