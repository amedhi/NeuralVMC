/*---------------------------------------------------------------------------
* @Author: Amal Medhi
* @Date:   2018-12-29 20:39:14
* @Last Modified by:   Amal Medhi
* @Last Modified time: 2024-02-20 15:35:33
*----------------------------------------------------------------------------*/
#include <boost/filesystem.hpp>
#include <filesystem>
#include "ffnet.h"

namespace ann {

FFNet::FFNet() : AbstractNet(), num_layers_{0}, num_params_{0}
{
  layers_.clear();
  num_params_fwd_.clear(); 
}

int FFNet::add_layer(const int& units, const std::string& activation, 
  const int& input_dim)
{
  if (layers_.size()==0) {
    if (input_dim<1) {
      throw std::invalid_argument("FFNet::add_layer: invalid value for 'input_dim'");
    }
    else {
      // input layer
      layers_.push_back(new NeuralLayer(input_dim,"None",input_dim));
      // first layer
      layers_.push_back(new NeuralLayer(units,activation,input_dim));
      layers_[1]->set_input_layer(layers_[0]);
      layers_[0]->set_output_layer(layers_[1]);
      return layers_.size()-1;
    }
  }
  // other layers
  if (input_dim==0 || layers_.back()->num_units()==input_dim) {
    layers_.push_back(new NeuralLayer(units,activation,layers_.back()->num_units()));
    for (int n=1; n<layers_.size(); ++n) {
      layers_[n]->set_input_layer(layers_[n-1]);
      layers_[n-1]->set_output_layer(layers_[n]);
    }
    return layers_.size()-1;
  }
  else {
    throw std::invalid_argument("FFNet::add_layer: invalid value for 'input_dim'");
  }
}

int FFNet::add_sign_layer(const int& input_dim)
{
  if (layers_.size()==0) {
    if (input_dim<1) {
      throw std::invalid_argument("FFNet::add_sign_layer: invalid value for 'input_dim'");
    }
    else {
      // input layer
      layers_.push_back(new NeuralLayer(input_dim,"None",input_dim));
      // sign layer
      layers_.push_back(new SignLayer("None",input_dim));
      layers_[1]->set_input_layer(layers_[0]);
      layers_[0]->set_output_layer(layers_[1]);
      return layers_.size()-1;
    }
  }
  // other layers
  if (input_dim==0 || layers_.back()->num_units()==input_dim) {
    layers_.push_back(new SignLayer("None",layers_.back()->num_units()));
    for (int n=1; n<layers_.size(); ++n) {
      layers_[n]->set_input_layer(layers_[n-1]);
      layers_[n-1]->set_output_layer(layers_[n]);
    }
    return layers_.size()-1;
  }
  else {
    throw std::invalid_argument("FFNet::add_sign_layer: invalid value for 'input_dim'");
  }
}

int FFNet::compile(void)
{
  num_layers_ = layers_.size(); // including input layer
  num_params_fwd_.resize(num_layers_);
  for (int i=0; i<num_layers_; ++i) {
    num_params_fwd_[0] = 0;
    layers_[i]->set_id(i);
  }
  for (int i=1; i<num_layers_; ++i) {
    int n = num_params_fwd_[i-1] + layers_[i]->num_params();
    num_params_fwd_[i] = n;
    //std::cout << "i  n ="<<i<<"  "<<n<<"\n";
  }
  if (num_layers_>0) num_params_ = num_params_fwd_.back();
  else num_params_ = 0;
  //output_.resize(layers_.back().num_units());
  input_changes_.resize(layers_.front()->num_units());
  //gradient_.resize(num_params_,layers_.back()->num_units());
  return 0;
}

void FFNet::init_parameters(random_engine& rng, const double& sigma) 
{
  for (auto& layer : layers_) layer->init_parameters(rng, sigma);
}

void FFNet::init_parameter_file(const std::string& save_path, const std::string& load_path)
{
  save_path_ = save_path;
  load_path_ = load_path;
}

void FFNet::save_parameters(void) const
{
  std::cout << "FFNet:: Saving parameters to file\n";
  boost::filesystem::path prefix_dir(save_path_);
  boost::filesystem::create_directories(prefix_dir);
  for (int n=1; n<num_layers_; ++n) {
    std::string fname = save_path_+"layer_"+std::to_string(n)+".txt";
    std::ofstream fs(fname);
    if (fs.is_open()) {
      layers_[n]->save_parameters(fs);
      fs.close();
    }
    else {
      throw std::range_error("FFNet::save_parameters: file open failed");
    }
  }
}

void FFNet::load_parameters(void)
{
  std::cout << "FFNet:: loading parameters from file\n";
  for (int n=1; n<num_layers_; ++n) {
    std::string fname = load_path_+"/layer_"+std::to_string(n)+".txt";
    std::ifstream fs(fname);
    if (fs.is_open()) {
      layers_[n]->load_parameters(fs);
      fs.close();
    }
    else {
      throw std::range_error("FFNet::load_parameters: file open failed");
    }
  }
}

const double& FFNet::get_parameter(const int& id) const
{
  for (int i=1; i<num_layers_; ++i) {
    if (id < num_params_fwd_[i]) {
      return layers_[i]->get_parameter(id-num_params_fwd_[i-1]);
    }
  }
  throw std::out_of_range("FFNet::get_parameter: out-of-range 'id'");
}

void FFNet::get_parameters(RealVector& pvec) const
{
  for (int i=1; i<num_layers_; ++i) {
    int pos = num_params_fwd_[i-1];
    layers_[i]->get_parameters(pvec, pos);
  }
}

void FFNet::get_parameter_names(std::vector<std::string>& pnames, const int& pos) const
{
  for (int i=1; i<num_layers_; ++i) {
    int start_pos = pos+num_params_fwd_[i-1];
    layers_[i]->get_parameter_names(pnames,start_pos);
  }
}

void FFNet::get_parameter_values(RealVector& pvalues, const int& pos) const
{
  for (int i=1; i<num_layers_; ++i) {
    int start_pos = pos+num_params_fwd_[i-1];
    layers_[i]->get_parameter_values(pvalues,start_pos);
  }
}

void FFNet::get_parameter_lbound(RealVector& lbound, const int& start_pos) const
{
  assert(lbound.size() >= start_pos+num_params_);
  for (int n=0; n<num_params_; ++n) {
    lbound[start_pos+n] = -std::numeric_limits<double>::infinity();
  }
}

void FFNet::get_parameter_ubound(RealVector& ubound, const int& start_pos) const
{
  assert(ubound.size() >= start_pos+num_params_);
  for (int n=0; n<num_params_; ++n) {
    ubound[start_pos+n] = std::numeric_limits<double>::infinity();
  }
}

void FFNet::update_parameters(const RealVector& pvec, const int& start_pos)
{
  for (int i=1; i<num_layers_; ++i) {
    int pos = start_pos + num_params_fwd_[i-1];
    layers_[i]->update_parameters(pvec, pos);
  }
}

void FFNet::update_parameter(const int& id, const double& value)
{
  for (int i=1; i<num_layers_; ++i) {
    if (id < num_params_fwd_[i]) {
      layers_[i]->update_parameter(id-num_params_fwd_[i-1], value);
      return;
    }
  }
  throw std::out_of_range("FFNet::update_parameter: out-of-range 'id'");
}

void FFNet::do_update_run(const RealVector& input)
{
  layers_.front()->update_forward(input);
  //std::cout<<"output = "<<layers_.back()->output()<< "\n"; getchar();
}

void FFNet::do_update_run(const RealVector& new_input, const std::vector<int> new_elems) 
{
  for (const auto& i : new_elems) {
    input_changes_(i) = new_input(i)-layers_.front()->output()(i);
  }
  layers_.front()->update_forward(new_input,new_elems,input_changes_);
}

RealVector FFNet::get_new_output(const RealVector& input) const
{
  /* does not change the state */
  //return layers_.back().get_new_output(input);
  layers_.front()->feed_forward(input);
  return layers_.back()->new_output();
}

RealVector FFNet::get_new_output(const RealVector& new_input, const std::vector<int> new_elems) const
{
  for (const auto& i : new_elems) {
    input_changes_(i) = new_input(i)-layers_.front()->output()(i);
  }
  layers_.front()->feed_forward(new_input,new_elems,input_changes_);
  return layers_.back()->new_output();
}

void FFNet::get_gradient(RealMatrix& gradient) const
{
  // derivative by 'back propagation' method
  layers_.back()->derivative(gradient, num_params_);

  /*
  // derivative by 'forward propagation' - highly Inefficient
  int n=0;
  for (int i=1; i<num_layers_; ++i) {
    for (int id=num_params_fwd_[i-1]; id<num_params_fwd_[i]; ++id) {
      int pid = id-num_params_fwd_[i-1];
      //std::cout << "i  pid = " << i << "  " << pid << "\n";
      Vector df = layers_.back().derivative_fwd(i,pid);
      gradient_.row(n++) = df.transpose();
    }
  }
  */
  /*
  // backpropagartion
  //double d = back().derivative(1,0)(0);
  for (int i=0; i<num_params_; ++i) {
    std::cout << "der["<<i<<"] = "<<gradient_.col(0)(i)<<"\t"<<fwd.col(0)(i)<<"\n";
  }
  getchar();
  */
}


void FFNet::get_log_gradient(RealMatrix& gradient) const
{
  throw std::range_error("FFNet::get_log_gradient: not implemented");
}




} // end namespace ann

