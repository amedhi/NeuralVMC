/*---------------------------------------------------------------------------
* @Author: Amal Medhi
* @Date:   2018-12-29 20:39:14
* @Last Modified by:   Amal Medhi, amedhi@mbpro
* @Last Modified time: 2019-01-13 10:04:01
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
      push_back(std::unique_ptr<DenseLayer>(new DenseLayer(units)));
      // first layer
      push_back(std::unique_ptr<DenseLayer>(new DenseLayer(units,activation,input_dim)));
      return size()-1;
    }
  }
  // other layers
  if (input_dim==0 || back()->num_units()==input_dim) {
    push_back(std::unique_ptr<DenseLayer>(
      new DenseLayer(units,activation,back()->num_units())));
    for (int n=1; n<size(); ++n) {
      operator[](n)->set_input_layer(operator[](n-1).get());
      operator[](n-1)->set_output_layer(operator[](n).get());
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
  int n = 0;
  for (int i=0; i<size(); ++i) {
    n += operator[](i)->num_params();
    pid_range_[i] = n;
  }
}

const double& SequentialNet::get_parameter(const int& id) const
{
  for (int i=1; i<num_layers_; ++i) {
    if (id < pid_range_[i]) {
      return operator[](i)->get_parameter(id-pid_range_[i-1]);
    }
  }
  throw std::out_of_range("SequentialNet::get_parameter: out-of-range 'id'");
}

void SequentialNet::update_parameter(const int& id, const double& value)
{
  for (int i=1; i<num_layers_; ++i) {
    if (id < pid_range_[i]) {
      operator[](i)->update_parameter(id-pid_range_[i-1], value);
      return;
    }
  }
  throw std::out_of_range("SequentialNet::update_parameter: out-of-range 'id'");
}

Vector SequentialNet::get_output(const Vector& input) {
  front()->set_input(input);
  //return back()->get_output();
  return back()->output();
}

/*
int NeuralNet::add_layer(const int& units, const std::string& activation, 
  const int& input_dim)
{
  //std::cout << "add_layer -- " << size() << "\n";
  // if first layer
  if (layers_.size()==0) {
  	if (input_dim<1) {
  	  throw std::invalid_argument("NeuralNet::add_layer: invalid value for 'input_dim'");
  	}
  	else {
  	  layers_.push_back(Layer(units,activation,input_dim));
      layers_[layers_.size()-1].name_ = std::to_string(layers_.size()-1);
  	  return layers_.size();
  	}
  }
  // other layers
  if (input_dim==0 || layers_.back().num_units()==input_dim) {
    layers_.push_back(Layer(units,activation,layers_.back().num_units()));
    layers_[layers_.size()-1].name_ = std::to_string(layers_.size()-1);
    for (int n=1; n<layers_.size(); ++n) {
      layers_[n].set_input_layer(&layers_[n-1]);
      layers_[n-1].set_output_layer(&layers_[n]);
    }
  	int n = layers_.size();
    //std::cout << n << " " << n-1 << " " << n-2 << "\n\n"; getchar();
  	//layers_[n-2].set_output_layer(&layers_[n-1]);
    //layers_[1].set_input_layer(&layers_[0]);
    //layers_[n-1].set_input_layer(&layers_[n-2]);
    //std::cout << "inlayer = "<< layers_[n-1].inlayer_->name_ << "\n";
  	return n;
  }
  else {
  	throw std::invalid_argument("NeuralNet::add_layer: invalid value for 'input_dim'");
  }
}

Vector NeuralNet::get_output(const Vector& input){
  //operator[](0).name_ = "first";
  //operator[](1).name_ = "second";
  //operator[](2).name_ = "third";
  layers_[0].set_input(input);
  return layers_.back().get_output();
}
*/

} // end namespace nnet

