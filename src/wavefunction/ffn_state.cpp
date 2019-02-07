/*---------------------------------------------------------------------------
* Copyright (C) 2015-2016 by Amal Medhi <amedhi@iisertvm.ac.in>.
* @Author: Amal Medhi, amedhi@mbpro
* @Date:   2019-01-29 12:56:31
* @Last Modified by:   Amal Medhi, amedhi@mbpro
* @Last Modified time: 2019-02-07 13:49:51
*----------------------------------------------------------------------------*/
#include "./ffn_state.h"

namespace var {

using namespace nnet;

FFN_State::FFN_State(const int& num_sites, const input::Parameters& inputs)
{
  init(num_sites, inputs);
}

int FFN_State::init(const int& num_sites, const input::Parameters& inputs)
{
  num_sites_ = num_sites;
  //set_particle_num(inputs);
  int num_units = 2*num_sites_;
  add_layer(num_units,"Sigmoid",num_units);
  add_layer(1,"Sigmoid");
  compile();
  //num_varparms_ = ffnet_.num_params();
  return 0;
}

void FFN_State::update_state(const eig::real_vec& fock_state)
{
  SequentialNet::run(fock_state);
}

const double& FFN_State::output(void) const
{
  return SequentialNet::output()(0);
}

double FFN_State::get_output(const eig::real_vec& input) const
{
  return SequentialNet::get_output(input)(0);
}

//double FFN_State::get_wf_amplitude(const eig::real_vec& fock_state)
//{
//}






} // end namespave var