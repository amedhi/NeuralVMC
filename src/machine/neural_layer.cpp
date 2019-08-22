/*---------------------------------------------------------------------------
* @Author: Amal Medhi
* @Date:   2018-12-29 12:01:09
* @Last Modified by:   Amal Medhi, amedhi@mbpro
* @Last Modified time: 2019-08-22 16:12:48
*----------------------------------------------------------------------------*/
#include <locale>
#include "neural_layer.h"

namespace ann {

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
  else if (fname=="TANH") {
    activation_.reset(new TANH());
  }
  else {
    throw std::invalid_argument("NeuralLayer:: undefined activation '"+fname+"'");
  }
  inlayer_ = nullptr;
  outlayer_ = nullptr;
  // initial kernel & bias
  //kernel_ = Matrix::Ones(num_units_,input_dim_);
  //bias_ = Vector::Zero(num_units_);
  kernel_ = Matrix::Random(num_units_,input_dim_);
  bias_ = Vector::Random(num_units_);
  //std::cout << num_units_ << " " << input_dim_ <<"\n"; getchar();
  //std::cout << "kernel_=\n" << kernel_ << "\n"; getchar();
  // other storages
  input_ = Vector::Zero(input_dim_);
  output_ = Vector::Zero(num_units_);
  lin_output_ = Vector::Zero(num_units_);
  lin_output_tmp_ = Vector::Zero(num_units_);
  output_changes_ = Vector::Zero(num_units_);
  der_activation_ = Vector::Zero(num_units_);
  der_backflow_ = Vector::Zero(input_dim_);
  derivative_ = Vector::Zero(num_units_);
  num_params_ = kernel_.size()+bias_.size();
}

void NeuralLayer::init_parameters(random_engine& rng, const double& sigma) 
{
  std::normal_distribution<double> random_normal(1.0,sigma);
  for (int i=0; i<kernel_.rows(); ++i) {
    for (int j=0; j<kernel_.cols(); ++j) {
      kernel_(i,j) = random_normal(rng);
    }
  }
  for (int i=0; i<bias_.size(); ++i) {
    bias_(i) = random_normal(rng);
  }
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

void NeuralLayer::get_parameter_names(std::vector<std::string>& pnames, const int& pos) const
{
  std::string w = "w"+std::to_string(id_)+"_";
  int n = pos;
  for (int i=0; i<kernel_.rows(); ++i) {
    for (int j=0; j<kernel_.cols(); ++j) {
      pnames[n] = w + std::to_string(i) + "," + std::to_string(j); 
      n++;
    }
  }
  std::string b = "b"+std::to_string(id_)+"_";
  for (int i=0; i<bias_.size(); ++i) {
    pnames[n++] = b + std::to_string(i);
  }
}

void NeuralLayer::get_parameter_values(eig::real_vec& pvalues, const int& pos) const
{
  for (int i=0; i<kernel_.size(); ++i)
    pvalues(pos+i) = *(kernel_.data()+i);
  int n = pos+kernel_.size();
  for (int i=0; i<bias_.size(); ++i)
    pvalues(n+i) = *(bias_.data()+i);
}

void NeuralLayer::get_parameters(Vector& pvec, const int& start_pos) const
{
  for (int i=0; i<kernel_.size(); ++i)
    pvec(start_pos+i) = *(kernel_.data()+i);
  int n = start_pos+kernel_.size();
  for (int i=0; i<bias_.size(); ++i)
    pvec(n+i) = *(bias_.data()+i);
}

void NeuralLayer::update_parameters(const Vector& pvec, const int& start_pos)
{
  for (int i=0; i<kernel_.size(); ++i)
    *(kernel_.data()+i) = pvec(start_pos+i);
  int n = start_pos+kernel_.size();
  for (int i=0; i<bias_.size(); ++i)
    *(bias_.data()+i) = pvec(n+i);
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

/*
Vector NeuralLayer::output(void) 
{
  if (inlayer_ == nullptr) {
    return input_;
  }
  else {
    //output_ = activation_.get()->function(kernel_*inlayer_->output()+bias_);
    lin_output_ = kernel_*inlayer_->output()+bias_;
    output_ = activation_.get()->function(lin_output_);
    return output_;
  } 
  //output_ = activation_.get()->function(kernel_*inlayer_->output()+bias_);
  //return output_;
}
*/

int NeuralLayer::update_forward(const Vector& input) 
{
  /* Update the state of each layer and propage forward */
  if (inlayer_ == nullptr) {
    output_ = input;
    outlayer_->update_forward(output_);
    return 0;
  }
  else {
    lin_output_.noalias() = kernel_*input+bias_;
    output_ = activation_.get()->function(lin_output_);
    if (outlayer_ == nullptr) return 0;
    outlayer_->update_forward(output_);
  } 
  return 0;
}

int NeuralLayer::update_forward(const Vector& new_input, 
  const std::vector<int>& new_elems, const Vector& input_changes) 
{
  /* Update the state of each layer and propagate forward */
  if (inlayer_ == nullptr) {
    // for the first layer, since output = input, 
    for (const auto& i : new_elems) {
      output_changes_(i) = input_changes(i);
      output_(i) = new_input(i);
    }
    outlayer_->update_forward(output_,new_elems,output_changes_);
    return 0;
  }
  else {
    // other layers
    if (new_elems.size()<new_input.size()) {
      // true for the first, hidden layer only
      for (const auto& i : new_elems)  {
        lin_output_.noalias() += kernel_.col(i) * input_changes(i);
      }
    }
    else lin_output_.noalias() = kernel_*new_input+bias_;
    output_ = activation_.get()->function(lin_output_);
    if (outlayer_ == nullptr) return 0;
    outlayer_->update_forward(output_);
  } 
  return 0;
}

Vector NeuralLayer::feed_forward(const Vector& input) const
{
  /* Feed forward without changing the state of the layer */
  if (inlayer_ == nullptr) {
    return outlayer_->feed_forward(input);
  }
  else {
    lin_output_tmp_.noalias() = kernel_*input+bias_;
    if (outlayer_ == nullptr) {
      return activation_.get()->function(lin_output_tmp_);
    }
    else {
      return outlayer_->feed_forward(activation_.get()->function(lin_output_tmp_));
    }
  } 
}

Vector NeuralLayer::get_new_output(const Vector& input) const
{
  /* Give output without changing the state of the layer */
  if (inlayer_ == nullptr) {
    return input;
  }
  else {
    lin_output_tmp_.noalias() = kernel_*inlayer_->get_new_output(input)+bias_;
    //std::cout << "lin_output = " << lin_output_tmp_.transpose() << "\n";
    return activation_.get()->function(lin_output_tmp_);
  }
}

int NeuralLayer::derivative(Matrix& derivative, const int& num_total_params) const
{
  if (inlayer_ == nullptr) return 0;
  // check sizes
  if (derivative.rows()!=num_total_params || derivative.cols()!=num_units_) {
    throw std::out_of_range("NeuralLayer::derivative: dimension mismatch");
  }
  // derivative of each output units
  int pid_begin = num_total_params-num_params_;
  der_activation_ = activation_.get()->derivative(lin_output_);
  for (int q=0; q<num_units_; ++q) {
    int p = pid_begin;
    // derivaive wrt weight parameters in this layer
    /* since parameters are COUNTED COLUMN wise */
    for (int n=0; n<kernel_.cols(); ++n) {
      for (int m=0; m<kernel_.rows(); ++m) {
        if (m == q) {
          derivative(p++,q) = der_activation_(m)*inlayer_->output()(n);
        }
        else {
          derivative(p++,q) = 0.0;
        }
      }
    }
    // derivaive wrt bias parameters in this layer
    for (int m=0; m<bias_.size(); ++m) {
      if (m == q) derivative(p++, q) = der_activation_(m);
      else derivative(p++, q) = 0.0;
    }
    // derivative wrt parameters in the previous layer
    for (int n=0; n<kernel_.cols(); ++n) {
      der_backflow_(n) = der_activation_(q) * kernel_(q,n);
    }
    return inlayer_->back_propagate(pid_begin, der_backflow_, derivative, q);
  }
  return 0;
}

int NeuralLayer::back_propagate(const int& pid_end, const RowVector& backflow,
  Matrix& derivative, const int& use_col) const
{
  if (inlayer_ == nullptr) return 0;
  // derivative of activation function wrt current 'linear output'
  der_activation_ = activation_.get()->derivative(lin_output_);
  // If not the output layer, multiply it by backflow values 
  // for each 'neuron' units
  if (outlayer_ != nullptr) {
    for (int i=0; i<num_units_; ++i) {
      der_activation_(i) *= backflow(i);
    }
  }
  int pid_begin = pid_end-num_params_;
  int p = pid_begin;
  // derivaive wrt weight parameters
  /* since parameters are COUNTED COLUMN wise */
  for (int n=0; n<kernel_.cols(); ++n) {
    for (int m=0; m<kernel_.rows(); ++m) {
      derivative(p++,use_col) = der_activation_(m)*inlayer_->output()(n);
    }
  }
  // derivaive wrt bias parameters
  for (int m=0; m<bias_.size(); ++m) {
    derivative(p++,use_col) = der_activation_(m);
  }
  // new backflow for the inner layer
  der_backflow_ = der_activation_.transpose() * kernel_;
  return inlayer_->back_propagate(pid_begin, der_backflow_, derivative, use_col);
}


Vector NeuralLayer::derivative_fwd(const int& lid, const int& pid) const
{
  assert(lid > 0); // input layer has no parameter

  if (lid < id_) {
    // parameter in previous layer
    derivative_ = kernel_ * inlayer_->derivative_fwd(lid, pid);
    for (int i=0; i<num_units_; ++i) {
      //derivative_(i) *= activation_.get()->derivative(output_(i));  
      derivative_(i) *= activation_.get()->derivative(lin_output_(i));  
    }
    //std::cout << "layer = " << id_ << "\n";
    //getchar();
    return derivative_;
  }
  else if (lid == id_) {
    // parameter in this layer
    if (pid < kernel_.size()) {
      derivative_.setZero();
      // column index of kernel-matrix
      int j = pid / num_units_; 
      double xj = inlayer_->output()(j); // from 'input' layer
      //std::cout << "j = "<<j<<"\n";
      //std::cout << inlayer_->output().transpose() <<"\n";
      //std::cout << "xj = "<<xj<<"\n";
      //getchar();
      // row index of kernel-matrix
      int i = pid % num_units_;
      derivative_(i) = activation_.get()->derivative(lin_output_(i)) * xj;  
      /*for (int i=0; i<num_units_; ++i) {
        derivative_(i) = activation_.get()->derivative(lin_output_(i)) * xj;  
      }*/
      //std::cout << "do = " << derivative_.transpose() << "\n";
      return derivative_;
    }
    else if (pid < num_params_) {
      derivative_.setZero();
      // row index of kernel-matrix
      int i = pid % num_units_;
      derivative_(i) = activation_.get()->derivative(lin_output_(i));
      //derivative_ = activation_.get()->derivative(output_);  
      return derivative_;
    }
    else {
      throw std::out_of_range("NeuralLayer::derivative: out-of-range 'pid'");
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




} // end namespace ann
