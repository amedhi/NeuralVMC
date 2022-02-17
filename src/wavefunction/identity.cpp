/*---------------------------------------------------------------------------
* @Author: Amal Medhi, amedhi@mbpro
* @Date:   2019-02-07 12:31:24
* @Last Modified by:   Amal Medhi
* @Last Modified time: 2022-02-13 12:36:37
*----------------------------------------------------------------------------*/
#include "./identity.h"

namespace var {

Identity::Identity(const MF_Order::order_t& order, const input::Parameters& inputs, 
    const lattice::LatticeGraph& graph) 
  : GroundState(order, MF_Order::pairing_t::null)
{
  init(inputs, graph);
}

int Identity::init(const input::Parameters& inputs, 
  const lattice::LatticeGraph& graph)
{
  name_ = "Identity";
  order_name_ = "Identity";
  // sites & bonds
  num_sites_ = graph.num_sites();
  num_bonds_ = graph.num_bonds();
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