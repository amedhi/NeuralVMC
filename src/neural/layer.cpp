/*---------------------------------------------------------------------------
* @Author: Amal Medhi
* @Date:   2018-12-29 12:01:09
* @Last Modified by:   Amal Medhi, amedhi@mbpro
* @Last Modified time: 2019-01-14 14:53:40
*----------------------------------------------------------------------------*/
#include <locale>
#include "layer.h"

namespace nnet {

NeuralLayer::NeuralLayer(const int& units, const std::string& activation, 
  const int& input_dim)
  : num_units_{units}, input_dim_{input_dim}
{
  // activation function
  std::locale loc;
  std::string fname=activation;
  for (auto& x : fname)
    x = std::toupper(x,loc);
  //std::cout << "activation = " << fname << "\n";
  if (fname=="NONE") {
    activation_.reset(new None());
  }
  else if (fname=="RELU") {
    activation_.reset(new RELU());
  }
  else if (fname=="SIGMOID") {
    activation_.reset(new Sigmoid());
  }
  else {
    throw std::invalid_argument("NeuralLayer:: undefined activation '"+fname+"'");
  }
  inlayer_ = nullptr;
  outlayer_ = nullptr;
  // initial kernel & bias
  input_ = Vector::Zero(num_units_);
  kernel_ = Matrix::Ones(num_units_,input_dim_);
  bias_ = Vector::Zero(num_units_);
  output_ = Vector::Zero(num_units_);
  derivative_ = Vector::Zero(num_units_);
  num_params_ = kernel_.size()+bias_.size();
}

const double& NeuralLayer::get_parameter(const int& id) const
{
  if (id < kernel_.size()) {
    return *(kernel_.data()+id);
  }
  else if (id < num_params_) {
    int n = id-kernel_.size();
    return *(bias_.data()+n);
  }
  else {
    throw std::out_of_range("NeuralLayer::get_parameter: out-of-range 'id'");
  }
}

void NeuralLayer::update_parameter(const int& id, const double& value)
{
  if (id < kernel_.size()) {
    *(kernel_.data()+id) = value;
  }
  else if (id < num_params_) {
    int n = id-kernel_.size();
    *(bias_.data()+n) = value;
  }
  else {
    throw std::out_of_range("NeuralLayer::get_parameter: out-of-range 'id'");
  }
}

Vector NeuralLayer::output(void) 
{
  if (inlayer_ == nullptr) {
    return input_;
  }
  else {
    output_ = activation_.get()->function(kernel_*inlayer_->output()+bias_);
    return output_;
  } 
  //output_ = activation_.get()->function(kernel_*inlayer_->output()+bias_);
  //return output_;
}

Vector NeuralLayer::derivative(const int& lid, const int& pid)
{
  assert(lid > 0); // input layer has no parameter

  if (lid < id_) {
    // parameter in previous layer
    derivative_ = kernel_ * inlayer_->derivative(lid, pid);
    for (int i=0; i<num_units_; ++i) {
      derivative_(i) *= activation_.get()->derivative(output_(i));  
    }
    return derivative_;
  }
  else if (lid == id_) {
    // parameter in this layer
    if (pid < kernel_.size()) {
      int j = pid / num_units_; // column index of kernel-matrix
      //std::cout << "pid = "<<pid<<"\n";
      //std::cout << "j = "<<j<<"\n";
      double xj = inlayer_->output()(j); // from 'input' layer
      for (int i=0; i<num_units_; ++i) {
        derivative_(i) = activation_.get()->derivative(output_(i)) * xj;  
      }
      //std::cout << "do = " << derivative_.transpose() << "\n";
      return derivative_;
    }
    else if (pid < num_params_) {
      derivative_ = activation_.get()->derivative(output_);  
      return derivative_;
    }
    else {
      throw std::out_of_range("NeuralLayer::derivative: out-of-range 'id'");
    }
    return derivative_;
  }
  else {
    throw std::out_of_range("NeuralLayer::derivative: invalid parameter");
  }
}

/*
int AbstractLayer::num_layers_ = 0;

InputLayer::InputLayer(const int& units) 
{
  num_units_ = units;
  input_ = Vector::Zero(num_units_);
  output_ = Vector::Zero(num_units_);
  kernel_ = Matrix::Identity(num_units_,num_units_);
  bias_ = Vector::Zero(num_units_);
  outlayer_ = nullptr;
  num_params_ = 0;
  id_ = num_layers_++;
}

int Layer::num_layers_ = 0;
Layer::Layer(const int& units, const std::string& activation, const int& input_dim)
  : num_units_{units}
  , input_dim_{input_dim}
{
  // activation function
  std::locale loc;
  std::string fname=activation;
  for (auto& x : fname)
    x = std::toupper(x,loc);
  //std::cout << "activation = " << fname << "\n";
  if (fname=="NONE") {
    activation_.reset(new None());
  }
  else if (fname=="NONE") {
    activation_.reset(new RELU());
  }
  else {
    activation_.reset(new None());
  }

  inlayer_ = nullptr;
  outlayer_ = nullptr;

  // initial kernel & bias
  kernel_ = Matrix::Ones(num_units_,input_dim_);
  bias_ = Vector::Zero(num_units_);
  input_ = Vector::Zero(input_dim_);
  //id
  id_ = num_layers_++;
  //std::cout << "Layer created = " << id_ << "\n";
}
*/

/*
Layer::Layer(const Layer& layer)
{
  id_=num_layers_++;
  name_ = layer.name_;
  num_units_ = layer.num_units_;
  input_dim_ = layer.input_dim_;
  inlayer_ = layer.inlayer_;
  outlayer_ = layer.outlayer_;
  activation_ = layer.activation_;
  kernel_ =layer.kernel_;
  bias_ = layer.bias_; 
  input_ = layer.input_;
  //std::cout << "Layer copied = " << id_ << "\n";
}
*/

/*
void Layer::set_input_layer(Layer* inlayer)
{
  inlayer_ = inlayer;
}

void Layer::set_output_layer(Layer* outlayer)
{
  outlayer_ = outlayer;
}

Vector Layer::get_output(void) const
{
  if (inlayer_ == nullptr) {
    //std::cout << "layer = " << name_ << "\n";
    return kernel_ * input_ + bias_;
  }
  else {
    //std::cout << "layer = " << name_ << "\n";
    //std::cout << input_ << "\n";
    return kernel_ * inlayer_->get_output() + bias_;
  } 
}
*/




} // end namespace nnet
