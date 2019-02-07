/*---------------------------------------------------------------------------
* Copyright (C) 2015-2016 by Amal Medhi <amedhi@iisertvm.ac.in>.
* @Author: Amal Medhi, amedhi@mbpro
* @Date:   2019-01-29 12:56:31
* @Last Modified by:   Amal Medhi, amedhi@mbpro
* @Last Modified time: 2019-01-29 12:57:04
*----------------------------------------------------------------------------*/

#ifndef FFN_STATE_H
#define FFN_STATE_H

#include "./groundstate.h"
#include "../neural/neuralnet.h"

namespace var {
class FFN_State : public nnet::SequentialNet 
{
public:
  FFN_State() : nnet::SequentialNet() {}
  FFN_State(const int& num_sites, const input::Parameters& inputs); 
  ~FFN_State() {} 
  int init(const int& num_sites, const input::Parameters& inputs);
  void update(const var::parm_vector& pvector, const unsigned& start_pos=0); 
  void update_state(const eig::real_vec& fock_state);
  const double& output(void) const; 
  double get_output(const eig::real_vec& input) const;
private:
  int num_sites_{0};
};





} // end namespave var
#endif