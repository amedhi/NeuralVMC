/*---------------------------------------------------------------------------
* @Author: Amal Medhi
* @Date:   2018-12-29 20:39:14
* @Last Modified by:   Amal Medhi, amedhi@mbpro
* @Last Modified time: 2019-02-11 16:45:15
*----------------------------------------------------------------------------*/
#include "neuralnet.h"

namespace nnet {

int SequentialNet::add_layer(const int& units, const std::string& activation, 
  const int& input_dim)
{
  if (size()==0) {
    if (input_dim<1) {
      throw std::invalid_argument("SequentialNet::add_layer: invalid value for 'input_dim'");
    }
    else {
      // input layer
      //push_back(std::unique_ptr<InputLayer>(new InputLayer(units)));
      push_back(NeuralLayer(input_dim,"None",input_dim));
      // first layer
      push_back(NeuralLayer(units,activation,input_dim));
      operator[](1).set_input_layer(&operator[](0));
      operator[](0).set_output_layer(&operator[](1));
      return size()-1;
    }
  }
  // other layers
  if (input_dim==0 || back().num_units()==input_dim) {
    push_back(NeuralLayer(units,activation,back().num_units()));
    for (int n=1; n<size(); ++n) {
      operator[](n).set_input_layer(&operator[](n-1));
      operator[](n-1).set_output_layer(&operator[](n));
    }
    /*
    layers_[layers_.size()-1].name_ = std::to_string(layers_.size()-1);
    for (int n=1; n<layers_.size(); ++n) {
      layers_[n].set_input_layer(&layers_[n-1]);
      layers_[n-1].set_output_layer(&layers_[n]);
    }*/
    return size()-1;
  }
  else {
    throw std::invalid_argument("SequentialNet::add_layer: invalid value for 'input_dim'");
  }
}

void SequentialNet::compile(void)
{
  num_layers_ = size(); // including input layer
  pid_range_.resize(num_layers_);
  for (int i=0; i<size(); ++i) operator[](i).set_id(i);
  pid_range_[0] = 0;
  for (int i=1; i<size(); ++i) {
    int n = pid_range_[i-1] + operator[](i).num_params();
    pid_range_[i] = n;
    //std::cout << "i  n ="<<i<<"  "<<n<<"\n";
  }
  if (num_layers_>0) num_params_ = pid_range_.back();
  else num_params_ = 0;
  output_.resize(back().num_units());
  gradient_.resize(num_params_,back().num_units());
}

const double& SequentialNet::get_parameter(const int& id) const
{
  for (int i=1; i<num_layers_; ++i) {
    if (id < pid_range_[i]) {
      return operator[](i).get_parameter(id-pid_range_[i-1]);
    }
  }
  throw std::out_of_range("SequentialNet::get_parameter: out-of-range 'id'");
}

void SequentialNet::get_parameters(Vector& pvec) const
{
  for (int i=1; i<num_layers_; ++i) {
    int pos = pid_range_[i-1];
    operator[](i).get_parameters(pvec, pos);
  }
}

void SequentialNet::get_parameter_names(std::vector<std::string>& pnames, const int& pos) const
{
  for (int i=1; i<num_layers_; ++i) {
    int start_pos = pos+pid_range_[i-1];
    operator[](i).get_parameter_names(pnames,start_pos);
  }
}

void SequentialNet::get_parameter_values(eig::real_vec& pvalues, const int& pos) const
{
  for (int i=1; i<num_layers_; ++i) {
    int start_pos = pos+pid_range_[i-1];
    operator[](i).get_parameter_values(pvalues,start_pos);
  }
}

void SequentialNet::update_parameters(const Vector& pvec)
{
  for (int i=1; i<num_layers_; ++i) {
    int pos = pid_range_[i-1];
    operator[](i).update_parameters(pvec, pos);
  }
}

void SequentialNet::update_parameter(const int& id, const double& value)
{
  for (int i=1; i<num_layers_; ++i) {
    if (id < pid_range_[i]) {
      operator[](i).update_parameter(id-pid_range_[i-1], value);
      return;
    }
  }
  throw std::out_of_range("SequentialNet::update_parameter: out-of-range 'id'");
}

eig::real_vec SequentialNet::get_output(const eig::real_vec& input) const
{
  /* does not change the state */
  return back().get_output(input);
}

void SequentialNet::run(const eig::real_vec& input)
{
  front().set_input(input);
  output_ = back().output();
}

const Matrix& SequentialNet::get_gradient(void) const
{
  int n=0;
  for (int i=1; i<num_layers_; ++i) {
    for (int id=pid_range_[i-1]; id<pid_range_[i]; ++id) {
      int pid = id-pid_range_[i-1];
      //std::cout << "i  pid = " << i << "  " << pid << "\n";
      Vector df = back().derivative(i,pid);
      gradient_.row(n++) = df.transpose();
    }
  }
  //double d = back().derivative(1,0)(0);
  //std::cout << "d = " << d << "\n";
  return gradient_;
}

} // end namespace nnet

