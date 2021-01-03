/*---------------------------------------------------------------------------
* Copyright (C) 2019 by Amal Medhi <amedhi@iisertvm.ac.in>.
* @Author: Amal Medhi, amedhi@mbpro
* @Date:   2019-08-13 12:00:53
* @Last Modified by:   Amal Medhi, amedhi@mbpro
* @Last Modified time: 2020-06-08 19:45:35
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
  //exponential_type_ = true;
  exponential_type_ = false;
  num_sites_ = num_sites;
  int num_units = 2*num_sites_;

  std::locale loc;
  have_sign_nnet_ = false;
  name_ = inputs.set_value("nqs_wf", "NONE");
  for (auto& x : name_) x = std::toupper(x,loc);
  if (name_ == "FFNN") {
    nnet_.reset(new ann::FFNet());
  	//nnet_->add_layer(num_units,"Lcosh",num_units);
    //nnet_->add_layer(num_units,"tanh",num_units);
    //nnet_->add_layer(num_units,"relu",num_units);
    nnet_->add_layer(num_units,"sigmoid",num_units);
    //nnet_->add_layer(num_units,"tanh",num_units);
  	//nnet_->add_layer(num_units,"tanh",num_units);
    //nnet_->add_layer(num_units,"Sigmoid",num_units);
  	nnet_->add_layer(1,"sigmoid");
    //nnet_->add_layer(1,"relu");
  	nnet_->compile();
    num_params_ = nnet_->num_params();
  }

/*
  else if (name_ == "FFNN_SIGN") {
    nnet_.reset(new ann::FFNet());
    //nnet_->add_layer(num_units,"Lcosh",num_units);
    //nnet_->add_layer(num_units,"tanh",num_units);
    nnet_->add_layer(num_units,"sigmoid",num_units);
    nnet_->add_layer(1,"relu");
    nnet_->compile();
    // sign part
    sign_nnet_.reset(new ann::FFNet());
    sign_nnet_->add_sign_layer(num_units);
    sign_nnet_->compile();
    have_sign_nnet_ = true;
    num_params_ = nnet_->num_params() + sign_nnet_->num_params();
  }
*/
  
 // /*
  else if (name_ == "FFNN_SIGN") {
    nnet_.reset(new ann::FFNet());
    //nnet_->add_layer(num_units,"Lcosh",num_units);
    //nnet_->add_layer(num_units,"tanh",num_units);
    nnet_->add_layer(num_units,"sigmoid",num_units);
    nnet_->add_layer(1,"sigmoid");
    nnet_->compile();
    // sign part
    sign_nnet_.reset(new ann::FFNet());
    sign_nnet_->add_layer(1,"cospi",num_units);
    sign_nnet_->compile();
    have_sign_nnet_ = true;
    num_params_ = nnet_->num_params() + sign_nnet_->num_params();
  }
  //*/

  else if (name_ == "SYMFFNN") {
    nnet_.reset(new ann::SymmFFNet());
    nnet_->add_layer(num_units,"tanh",num_units);
    //nnet_->add_layer(num_units,"Sigmoid",num_units);
    nnet_->add_layer(1,"sigmoid");
    nnet_->compile();
    num_params_ = nnet_->num_params();
  }

  else {
    throw std::range_error("NQS_Wavefunction::NQS_Wavefunction: unidefined neural net");
  }
  return 0;
}

const int& NQS_Wavefunction::num_params(void) const 
{ 
	return num_params_;
}

void NQS_Wavefunction::init_parameters(ann::random_engine& rng, const double& sigma)
{
  nnet_->init_parameters(rng,sigma); 
  if (have_sign_nnet_) sign_nnet_->init_parameters(rng,sigma); 
}

void NQS_Wavefunction::get_parm_names(std::vector<std::string>& pnames, const int& pos) const
{
	nnet_->get_parameter_names(pnames, pos);
  if (have_sign_nnet_) 
    sign_nnet_->get_parameter_names(pnames,pos+nnet_->num_params());
}

void NQS_Wavefunction::get_parm_values(ann::Vector& pvalues, const int& pos) const
{
	nnet_->get_parameter_values(pvalues, pos);
  if (have_sign_nnet_) 
    sign_nnet_->get_parameter_values(pvalues, pos+nnet_->num_params());
}

void NQS_Wavefunction::get_parm_vector(std::vector<double>& pvalues, const int& pos) const
{
  nnet_->get_parameter_vector(pvalues, pos);
  if (have_sign_nnet_) 
    sign_nnet_->get_parameter_vector(pvalues, pos+nnet_->num_params());
}

void NQS_Wavefunction::update_parameters(const ann::Vector& pvalues, const int& pos)
{
  nnet_->update_parameters(pvalues,pos);
  if (have_sign_nnet_) 
    sign_nnet_->update_parameters(pvalues,pos+nnet_->num_params());
}

void NQS_Wavefunction::update_state(const ann::ivector& fock_state)
{
  nnet_->do_update_run(fock_state.cast<double>());
  if (have_sign_nnet_) 
    sign_nnet_->do_update_run(fock_state.cast<double>());
}

void NQS_Wavefunction::update_state(const ann::ivector& fock_state, 
  const std::vector<int> new_elems)
{
  nnet_->do_update_run(fock_state.cast<double>(), new_elems);
  if (have_sign_nnet_) 
    sign_nnet_->do_update_run(fock_state.cast<double>(), new_elems);
}

const amplitude_t& NQS_Wavefunction::output(void) const
{
  if (exponential_type_) {
    if (have_sign_nnet_) {
      output_ = std::exp(nnet_->output()(0) + ii()*sign_nnet_->output()(0));
      std::cout << "sign net = "<<sign_nnet_->output()(0)/3.14159265358979323846 <<"\n";
    }
    else {
      output_ = std::exp(nnet_->output()(0));
    }
  }
  else {
    if (have_sign_nnet_) {
      output_ = nnet_->output()(0) * sign_nnet_->output()(0);
    }
    else {
      output_ = nnet_->output()(0);
    }
  }
  return output_;
}

amplitude_t NQS_Wavefunction::get_new_output(const ann::ivector& fock_state) const
{
  if (exponential_type_) {
    if (have_sign_nnet_) {
      return std::exp(nnet_->get_new_output(fock_state.cast<double>())(0) 
             + ii()*sign_nnet_->get_new_output(fock_state.cast<double>())(0));
    }
    else {
      return std::exp(nnet_->get_new_output(fock_state.cast<double>())(0));
    }
  }
  else {
    if (have_sign_nnet_) {
      return nnet_->get_new_output(fock_state.cast<double>())(0) 
             * sign_nnet_->get_new_output(fock_state.cast<double>())(0);
    }
    else {
      return nnet_->get_new_output(fock_state.cast<double>())(0);
    }
  }
}

amplitude_t NQS_Wavefunction::get_new_output(const ann::ivector& fock_state, 
  const std::vector<int> new_elems) const
{
  if (exponential_type_) {
    if (have_sign_nnet_) {
      return std::exp(nnet_->get_new_output(fock_state.cast<double>(), new_elems)(0)
            + ii()*sign_nnet_->get_new_output(fock_state.cast<double>(), new_elems)(0));
    }
    else {
      return std::exp(nnet_->get_new_output(fock_state.cast<double>(), new_elems)(0));
    }
  }
  else {
    if (have_sign_nnet_) {
      return nnet_->get_new_output(fock_state.cast<double>(), new_elems)(0)
           * sign_nnet_->get_new_output(fock_state.cast<double>(), new_elems)(0);
    }
    else {
     return nnet_->get_new_output(fock_state.cast<double>(), new_elems)(0);
    }
  }
}

void NQS_Wavefunction::get_gradient(Vector& grad, const int& pos) const
{
  if (exponential_type_) {
    if (have_sign_nnet_) {
      grad.head(nnet_->num_params()) = output_ * nnet_->get_gradient().col(0).cast<amplitude_t>();
      grad.tail(sign_nnet_->num_params()) = output_ * sign_nnet_->get_gradient().col(0).cast<amplitude_t>();
    }
    else {
      grad = output_ * nnet_->get_gradient().col(0).cast<amplitude_t>();
    }
  }
  else {
    if (have_sign_nnet_) {
      grad.head(nnet_->num_params()) = sign_nnet_->output()(0)*nnet_->get_gradient().col(0).cast<amplitude_t>();
      grad.tail(sign_nnet_->num_params()) = nnet_->output()(0)*sign_nnet_->get_gradient().col(0).cast<amplitude_t>();
    }
    else {
      grad = nnet_->get_gradient().col(0).cast<amplitude_t>();
    }
  }
}

void NQS_Wavefunction::get_log_gradient(Vector& grad, const int& pos) const
{
  if (exponential_type_) {
    if (have_sign_nnet_) {
      grad.head(nnet_->num_params()) = nnet_->get_gradient().col(0).cast<amplitude_t>();
      grad.tail(sign_nnet_->num_params()) = ii()*sign_nnet_->get_gradient().col(0).cast<amplitude_t>();
    }
    else {
      grad = nnet_->get_gradient().col(0).cast<amplitude_t>();
    }
  }
  else {
    amplitude_t inv = ampl_part(1.0)/output_;
    if (have_sign_nnet_) {
      grad.head(nnet_->num_params()) = inv*sign_nnet_->output()(0)*nnet_->get_gradient().col(0).cast<amplitude_t>();
      grad.tail(sign_nnet_->num_params()) = inv*nnet_->output()(0)*sign_nnet_->get_gradient().col(0).cast<amplitude_t>();
    }
    else {
      grad = inv*nnet_->get_gradient().col(0).cast<amplitude_t>();
    }
  }
}


} // end namespace ML







