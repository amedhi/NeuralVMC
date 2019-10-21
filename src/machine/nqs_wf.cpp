/*---------------------------------------------------------------------------
* Copyright (C) 2019 by Amal Medhi <amedhi@iisertvm.ac.in>.
* @Author: Amal Medhi, amedhi@mbpro
* @Date:   2019-08-13 12:00:53
* @Last Modified by:   Amal Medhi, amedhi@mbpro
* @Last Modified time: 2019-10-21 11:21:06
*----------------------------------------------------------------------------*/
#include <locale>
#include "nqs_wf.h"

namespace nqs {

NQS_Wavefunction::NQS_Wavefunction(const int& num_sites, const input::Parameters& inputs)
{
  init(num_sites, inputs);
}

int NQS_Wavefunction::init(const int& num_sites, const input::Parameters& inputs)
{
  num_sites_ = num_sites;
  int num_units = 2*num_sites_;

  std::locale loc;
  name_ = inputs.set_value("nqs_wf", "NONE");
  for (auto& x : name_) x = std::toupper(x,loc);
  if (name_ == "FFNN") {
    nnet_.reset(new ann::FFNet());
  	nnet_->add_layer(num_units,"tanh",num_units);
  	//nnet_->add_layer(num_units,"tanh",num_units);
    //nnet_->add_layer(num_units,"Sigmoid",num_units);
  	nnet_->add_layer(1,"sigmoid");
  	nnet_->compile();
  }
  else {
    throw std::range_error("NQS_Wavefunction::NQS_Wavefunction: unidefined neural net");
  }
  return 0;
}

const int& NQS_Wavefunction::num_params(void) const 
{ 
	return nnet_->num_params(); 
}

void NQS_Wavefunction::init_parameters(ann::random_engine& rng, const double& sigma)
{
  nnet_->init_parameters(rng, sigma); 
}

void NQS_Wavefunction::get_parm_names(std::vector<std::string>& pnames, const int& pos) const
{
	nnet_->get_parameter_names(pnames, pos);
}

void NQS_Wavefunction::get_parm_values(ann::Vector& pvalues, const int& pos) const
{
	nnet_->get_parameter_values(pvalues, pos);
}

void NQS_Wavefunction::get_parm_vector(std::vector<double>& pvalues, const int& pos) const
{
  nnet_->get_parameter_vector(pvalues, pos);
}

void NQS_Wavefunction::update_parameters(const ann::Vector& pvalues, const int& pos)
{
  nnet_->update_parameters(pvalues,pos);
}

void NQS_Wavefunction::update_state(const ann::ivector& fock_state)
{
  nnet_->do_update_run(fock_state.cast<double>());
}

void NQS_Wavefunction::update_state(const ann::ivector& fock_state, 
  const std::vector<int> new_elems)
{
  nnet_->do_update_run(fock_state.cast<double>(), new_elems);
}

const double& NQS_Wavefunction::output(void) const
{
  return nnet_->output()(0);
}

double NQS_Wavefunction::get_new_output(const ann::ivector& fock_state) const
{
  return nnet_->get_new_output(fock_state.cast<double>())(0);
}

double NQS_Wavefunction::get_new_output(const ann::ivector& fock_state, 
  const std::vector<int> new_elems) const
{
  return nnet_->get_new_output(fock_state.cast<double>(), new_elems)(0);
}

void NQS_Wavefunction::get_gradient(ann::Vector& grad, const int& pos) const
{
  grad = nnet_->get_gradient().col(0);
}





} // end namespace ML







