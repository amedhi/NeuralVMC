/*---------------------------------------------------------------------------
* @Author: Amal Medhi, amedhi@mbpro
* @Date:   2019-02-07 12:31:24
* @Last Modified by:   Amal Medhi, amedhi@mbpro
* @Last Modified time: 2019-02-07 12:53:34
*----------------------------------------------------------------------------*/
#include "./identity.h"

namespace var {

Identity::Identity(const input::Parameters& inputs, 
    const lattice::LatticeGraph& graph) 
  : GroundState(true)
{
  init(inputs, graph);
}

int Identity::init(const input::Parameters& inputs, 
  const lattice::LatticeGraph& graph)
{
  // sites & bonds
  num_sites_ = graph.num_sites();
  num_bonds_ = graph.num_bonds();
  // particle number
  set_particle_num(inputs);
  varparms_.clear();
  num_varparms_ = 0;
  return 0;
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