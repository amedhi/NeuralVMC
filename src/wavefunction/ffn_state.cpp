/*---------------------------------------------------------------------------
* Copyright (C) 2015-2016 by Amal Medhi <amedhi@iisertvm.ac.in>.
* @Author: Amal Medhi, amedhi@mbpro
* @Date:   2019-01-29 12:56:31
* @Last Modified by:   Amal Medhi, amedhi@mbpro
* @Last Modified time: 2019-08-14 12:12:28
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
  add_layer(num_units,"tanh",num_units);
  add_layer(num_units,"tanh",num_units);
  add_layer(1,"sigmoid");
  compile();
  //num_varparms_ = ffnet_.num_params();
  return 0;
}

void FFN_State::update_state(const eig::ivector& fock_state)
{
  SequentialNet::run(fock_state.cast<double>());
}

void FFN_State::update_params(const eig::real_vec& pvalues, const int& pos)
{
  SequentialNet::update_parameters(pvalues, pos);
}

const double& FFN_State::output(void) const
{
  return SequentialNet::output()(0);
}

double FFN_State::get_output(const eig::ivector& input) const
{
  return SequentialNet::get_output(input.cast<double>())(0);
}

void FFN_State::get_parm_names(std::vector<std::string>& pnames,const int& pos) const
{
  SequentialNet::get_parameter_names(pnames, pos);
}

void FFN_State::get_parm_values(eig::real_vec& pvalues, const int& pos) const
{
  SequentialNet::get_parameter_values(pvalues, pos);
}

void FFN_State::get_parm_lbound(eig::real_vec& lbound, const int& pos) const
{
  for (int i=0; i<num_params(); ++i) lbound(pos+i) = -10.0;
}

void FFN_State::get_parm_ubound(eig::real_vec& ubound, const int& pos) const
{
  for (int i=0; i<num_params(); ++i) ubound(pos+i) = +10.0;
}

void FFN_State::get_parm_vector(std::vector<double>& pvalues, const int& pos) const
{
  eig::real_vec pvec(num_params());
  SequentialNet::get_parameter_values(pvec, 0);
  for (int i=0; i<num_params(); ++i) pvalues[pos+i] = pvec[i];
}

void FFN_State::get_gradient(eig::real_vec& grad, const int& pos) const
{
  grad = SequentialNet::get_gradient().col(0);
  //std::cout << grad.transpose() << "\n"; getchar();
}





} // end namespave var