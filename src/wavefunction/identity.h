/*---------------------------------------------------------------------------
* @Author: Amal Medhi, amedhi@mbpro
* @Date:   2019-02-07 12:31:24
* @Last Modified by:   Amal Medhi, amedhi@mbpro
* @Last Modified time: 2019-02-07 12:31:56
*----------------------------------------------------------------------------*/

#ifndef IDENTITY_H
#define IDENTITY_H

#include "./groundstate.h"

namespace var {

class Identity : public GroundState
{
public:
  Identity() : GroundState(true) {}
  Identity(const input::Parameters& inputs, const lattice::LatticeGraph& graph); 
  ~Identity() {} 
  int init(const input::Parameters& inputs, const lattice::LatticeGraph& graph);
  std::string info_str(void) const override; 
  void update(const input::Parameters& inputs) override;
  void update(const var::parm_vector& pvector, const unsigned& start_pos=0) override;
  void get_wf_amplitudes(Matrix& psi) override;
  void get_wf_gradient(std::vector<Matrix>& psi_gradient) override; 
private:
  std::string order_name_;
};


} // end namespace var


#endif