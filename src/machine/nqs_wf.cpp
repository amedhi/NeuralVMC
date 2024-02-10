/*---------------------------------------------------------------------------
* Copyright (C) 2019 by Amal Medhi <amedhi@iisertvm.ac.in>.
* @Author: Amal Medhi, amedhi@mbpro
* @Date:   2019-08-13 12:00:53
* @Last Modified by:   Amal Medhi
* @Last Modified time: 2024-02-10 11:36:43
*----------------------------------------------------------------------------*/
#include <locale>
#include "nqs_wf.h"

namespace nqs {

NQS_Wavefunction::NQS_Wavefunction(const lattice::Lattice& lattice, const input::Parameters& inputs)
{
  init(lattice, inputs);
}

int NQS_Wavefunction::init(const lattice::Lattice& lattice, const input::Parameters& inputs)
{
  exponential_type_ = false;
  num_sites_ = lattice.num_sites();
  int num_units = 2*num_sites_;

  std::locale loc;
  have_sign_nnet_ = false;
  name_ = inputs.set_value("nqs_wf", "NONE");
  for (auto& x : name_) x = std::toupper(x,loc);
  if (name_ == "FFNN") {
    nid_ = net_id::FFNN;
    nnet_.reset(new ann::FFNet());
  	//nnet_->add_layer(num_units,"Lcosh",num_units);
    //nnet_->add_layer(num_units,"tanh",num_units);
    //nnet_->add_layer(num_units,"relu",num_units);
    nnet_->add_layer(num_units,"sigmoid",num_units);
    //nnet_->add_layer(num_units,"tanh",num_units);
  	//nnet_->add_layer(num_units,"tanh",num_units);
    //nnet_->add_layer(num_units,"Sigmoid",num_units);
  	nnet_->add_layer(1,"sigmoid");
    //nnet_->add_layer(1,"tanh");
    //nnet_->add_layer(1,"relu");
  	nnet_->compile();
    num_params_ = nnet_->num_params();
    num_output_units_ = nnet_->num_output_units(); 
  }

/*
  else if (name_ == "FFNN_CMPL") {
    nnet_.reset(new ann::FFNet());
    nnet_->add_layer(num_units,"sigmoid",num_units);
    nnet_->add_layer(1,"shifted_sigmoid",num_units);
    nnet_->compile();
    num_params_ = nnet_->num_params();

    // imaginary part
    sign_nnet_.reset(new ann::FFNet());
    sign_nnet_->add_layer(num_units,"sigmoid",num_units);
    sign_nnet_->add_layer(1,"shifted_sigmoid",num_units);
    sign_nnet_->compile();

    have_sign_nnet_ = true;
    num_params_ = nnet_->num_params() + sign_nnet_->num_params();
  }
*/

 // /*
  else if (name_ == "FFNN_SIGN") {
    nid_ = net_id::FFNN_SIGN;
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
    num_output_units_ = nnet_->num_output_units(); // assuming 1 for sign_net
  }
  //*/

  else if (name_ == "RBM") {
    nid_ = net_id::RBM;
    nnet_.reset(new ann::RBM(lattice, inputs));
    num_params_ = nnet_->num_params();
    num_output_units_ = nnet_->num_output_units();
  }

  else {
    throw std::range_error("NQS_Wavefunction::NQS_Wavefunction: unidefined neural net");
  }

  // matrices
  gradient_mat_.resize(num_params_,num_output_units_);

  return 0;
}

void NQS_Wavefunction::init_parameter_file(const std::string& prefix)
{
  std::string nqs_dir = prefix+"/nqs";
  switch (nid_) {
    case net_id::FFNN:
      nnet_->init_parameter_file(nqs_dir+"/ffnn");
      break;
    case net_id::FFNN_SIGN:
      nnet_->init_parameter_file(nqs_dir+"/ffnn");
      sign_nnet_->init_parameter_file(nqs_dir+"/ffnn_sign");
      break;
    case net_id::RBM:
      nnet_->init_parameter_file(nqs_dir+"/rbm");
      break;
    default: 
      throw std::range_error("NQS_Wavefunction::init_parameter_file: unknown net_id");
  }
} 

void NQS_Wavefunction::save_parameters(void) const
{
  nnet_->save_parameters();
  if (nid_ == net_id::FFNN_SIGN) {
    sign_nnet_->save_parameters();
  }
}

void NQS_Wavefunction::load_parameters(const std::string& load_path) 
{
  switch (nid_) {
    case net_id::FFNN:
      nnet_->load_parameters(load_path+"/nqs"+"/ffnn");
      break;
    case net_id::FFNN_SIGN:
      nnet_->load_parameters(load_path+"/nqs"+"/ffnn");
      sign_nnet_->load_parameters(load_path+"/nqs"+"/ffnn_sign");
      break;
    case net_id::RBM:
      nnet_->load_parameters(load_path+"/nqs"+"/rbm");
      break;
    default: 
      throw std::range_error("NQS_Wavefunction::load_parameters: unknown net_id");
  }
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

void NQS_Wavefunction::get_parm_values(RealVector& pvalues, const int& pos) const
{
	nnet_->get_parameter_values(pvalues, pos);
  if (have_sign_nnet_) 
    sign_nnet_->get_parameter_values(pvalues, pos+nnet_->num_params());
}

void NQS_Wavefunction::update_parameters(const RealVector& pvalues, const int& pos)
{
  nnet_->update_parameters(pvalues,pos);
  if (have_sign_nnet_) 
    sign_nnet_->update_parameters(pvalues,pos+nnet_->num_params());
}

void NQS_Wavefunction::update_parameter(const int& id, const double& value)
{
  nnet_->update_parameter(id,value);
}

void NQS_Wavefunction::update_state(const IntVector& fock_state)
{
  nnet_->do_update_run(fock_state.cast<double>());
  if (have_sign_nnet_) 
    sign_nnet_->do_update_run(fock_state.cast<double>());
}

void NQS_Wavefunction::update_state(const IntVector& fock_state, 
  const std::vector<int> new_elems)
{
  nnet_->do_update_run(fock_state.cast<double>(), new_elems);
  if (have_sign_nnet_) 
    sign_nnet_->do_update_run(fock_state.cast<double>(), new_elems);
}

const amplitude_t& NQS_Wavefunction::output(void) const
{
  if (complex_type_) {
    //output_ = std::exp(nnet_->output()(0) + ii()*nnet_->output()(1));
    output_ = nnet_->output()(0) + ii()*nnet_->output()(1);
  }
  else if (exponential_type_) {
    if (have_sign_nnet_) {
      output_ = std::exp(nnet_->output()(0) + ii()*sign_nnet_->output()(0));
      //std::cout << "sign net = "<<sign_nnet_->output()(0)/3.14159265358979323846 <<"\n";
    }
    else {
      output_ = std::exp(nnet_->output()(0));
    }
  }
  else {
    if (have_sign_nnet_) {
      output_ = nnet_->output()(0) * sign_nnet_->output()(0);
      //output_ = nnet_->output()(0) + ii()*sign_nnet_->output()(0);
      //output_ = nnet_->output()(0) * std::exp(ii()*sign_nnet_->output()(0));
    }
    else {
      output_ = nnet_->output()(0);
      //std::cout << output_ << "\n"; getchar();
    }
  }
  return output_;
}

amplitude_t NQS_Wavefunction::get_new_output(const IntVector& fock_state) const
{
  if (complex_type_) {
    Vector v = nnet_->get_new_output(fock_state.cast<double>());
    return v(0)+ii()*v(1);
    //return std::exp(v(0)+ii()*v(1));
  }
  else if (exponential_type_) {
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

      //return nnet_->get_new_output(fock_state.cast<double>())(0) 
      //     + ii()*sign_nnet_->get_new_output(fock_state.cast<double>())(0);

      //return nnet_->get_new_output(fock_state.cast<double>())(0) * 
      //   std::exp(ii()*sign_nnet_->get_new_output(fock_state.cast<double>())(0));
    }
    else {
      return nnet_->get_new_output(fock_state.cast<double>())(0);
    }
  }
}

amplitude_t NQS_Wavefunction::get_new_output(const IntVector& fock_state, 
  const std::vector<int> new_elems) const
{
  if (complex_type_) {
    Vector v = nnet_->get_new_output(fock_state.cast<double>(),new_elems);
    return v(0) + ii()*v(1);
    //return std::exp(v(0) + ii()*v(1));
  }
  else if (exponential_type_) {
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

      //return nnet_->get_new_output(fock_state.cast<double>(), new_elems)(0) 
      //  + ii()*sign_nnet_->get_new_output(fock_state.cast<double>(), new_elems)(0);

      //return nnet_->get_new_output(fock_state.cast<double>(), new_elems)(0) *
      //   std::exp(ii()*sign_nnet_->get_new_output(fock_state.cast<double>(), new_elems)(0));
    }
    else {
     return nnet_->get_new_output(fock_state.cast<double>(), new_elems)(0);
    }
  }
}

void NQS_Wavefunction::get_gradient(Vector& grad, const int& pos) const
{
  switch (nid_) {
    case net_id::RBM:
      nnet_->get_gradient(gradient_mat_);
      break;
    default: 
      throw std::range_error("NQS_Wavefunction::get_gradient: not implemented for this NET");
  }
  // copy
  for (int n=0; n<num_params_; ++n) {
    grad[pos+n] = gradient_mat_(n,0);
  }
}

void NQS_Wavefunction::get_log_gradient(Vector& grad, const int& pos) const
{
  switch (nid_) {
    case net_id::RBM:
      nnet_->get_log_gradient(gradient_mat_);
      break;
    default: 
      throw std::range_error("NQS_Wavefunction::get_gradient: not implemented for this NET");
  }
  // copy
  for (int n=0; n<num_params_; ++n) {
    grad[pos+n] = gradient_mat_(n,0);
  }
}


} // end namespace ML







