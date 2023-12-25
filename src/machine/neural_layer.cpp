/*---------------------------------------------------------------------------
* @Author: Amal Medhi
* @Date:   2018-12-29 12:01:09
* @Last Modified by:   Amal Medhi
* @Last Modified time: 2022-07-11 16:59:13
*----------------------------------------------------------------------------*/
#include <locale>
#include <boost/tokenizer.hpp>
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
  else if (fname=="SHIFTED_SIGMOID") {
    activation_.reset(new ShiftedSigmoid());
  }
  else if (fname=="TANH") {
    activation_.reset(new TANH());
  }
  else if (fname=="LCOSH") {
    activation_.reset(new LCOSH());
  }
  else if (fname=="COSPI") {
    activation_.reset(new COSPI());
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
  output_tmp_ = Vector::Zero(num_units_);
  lin_output_tmp_ = Vector::Zero(num_units_);
  output_changes_ = Vector::Zero(num_units_);
  der_activation_ = Vector::Zero(num_units_);
  der_backflow_ = Vector::Zero(input_dim_);
  derivative_ = Vector::Zero(num_units_);
  num_params_ = kernel_.size()+bias_.size();
}

void NeuralLayer::init_parameters(random_engine& rng, const double& sigma) 
{
  //std::uniform_real_distribution<double> real_generator(-1.0,1.0);
  //std::normal_distribution<double> random_normal(1.0,sigma);
  std::normal_distribution<double> random_normal(0.0,sigma);
  for (int i=0; i<kernel_.rows(); ++i) {
    for (int j=0; j<kernel_.cols(); ++j) {
      kernel_(i,j) = random_normal(rng);
      //kernel_(i,j) = real_generator(rng);
    }
  }
  for (int i=0; i<bias_.size(); ++i) {
    bias_(i) = random_normal(rng);
    //bias_(i) = real_generator(rng);
  }
}

void NeuralLayer::save_parameters(std::ofstream& fout) const
{
  fout << "#  Neural layer: num_units = "<<num_units_<<" , input_dim = "<<input_dim_<<"\n";
  fout << "#  Bias         |   Kernel\n";
  fout << std::scientific << std::uppercase << std::setprecision(6) << std::right;
  for (int i=0; i<num_units_; ++i) {
    fout << std::setw(15)<< bias_(i) << "  ";
    for (int j=0; j<input_dim_; ++j) {
      fout << std::setw(15) << kernel_(i,j);
    }
    fout << "\n";
  }
} 

void NeuralLayer::load_parameters(std::ifstream& fin) 
{
  boost::char_separator<char> space(" ");
  boost::tokenizer<boost::char_separator<char> >::iterator it;
  std::string line;
  std::string::size_type pos;
  int row = 0;
  while (std::getline(fin,line)) {
    // skip comments & blank lines
    pos = line.find_first_of("#");
    if (pos != std::string::npos) line.erase(pos);
    if (line.find_first_not_of(" ") == std::string::npos) continue;
    boost::tokenizer<boost::char_separator<char> > tokens(line, space);
    if (std::distance(tokens.begin(), tokens.end()) != (1+input_dim_)) {
      throw std::range_error("NeuralLayer::load_parameters: incorrect number of columns\n");
    }
    it=tokens.begin();
    // load 
    bias_(row) = std::stod(*it);
    int col = 0;
    for (++it; it!=tokens.end(); ++it) {
      kernel_(row,col++) = std::stod(*it);
    }
    row++;
  }
  if (row != num_units_) {
    throw std::range_error("NeuralLayer::load_parameters: incorrect number of rows\n");
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
  //std::cout << "-------------update\n";
  //std::cout << "kernel =\n" << kernel_ << "\n\n\n"; getchar();
  //std::cout<< "kernel = "<<kernel_<<std::endl;
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

int NeuralLayer::feed_forward(const Vector& input) const
{
  /* Feed forward without changing the state of the layer */
  if (inlayer_ == nullptr) {
    output_tmp_ = input;
    outlayer_->feed_forward(output_tmp_);
    return 0;
  }
  else {
    lin_output_tmp_.noalias() = kernel_*input+bias_;
    output_tmp_ = activation_.get()->function(lin_output_tmp_);
    if (outlayer_ == nullptr) return 0;
    outlayer_->feed_forward(output_tmp_);
  } 
  return 0;
}

int NeuralLayer::feed_forward(const Vector& new_input, 
  const std::vector<int>& new_elems, const Vector& input_changes) const
{
  /* Feed forward without changing the state of the layer */
  if (inlayer_ == nullptr) {
    // for the first layer, since output = input, 
    for (const auto& i : new_elems) {
      output_changes_(i) = input_changes(i);
      output_tmp_(i) = new_input(i);
    }
    outlayer_->feed_forward(output_tmp_,new_elems,output_changes_);
    return 0;
  }
  else {
    // other layers
    if (new_elems.size()<new_input.size()) {
      // true for the first, hidden layer only
      lin_output_tmp_ = lin_output_;
      for (const auto& i : new_elems)  {
        lin_output_tmp_.noalias() += kernel_.col(i) * input_changes(i);
      }
    }
    else lin_output_tmp_.noalias() = kernel_*new_input+bias_;
    output_tmp_ = activation_.get()->function(lin_output_tmp_);
    if (outlayer_ == nullptr) return 0;
    outlayer_->feed_forward(output_tmp_);
  } 
  return 0;
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
    } else { throw std::out_of_range("NeuralLayer::derivative: out-of-range 'pid'");
    }
    return derivative_;
  }
  else {
    throw std::out_of_range("NeuralLayer::derivative: invalid parameter");
  }
}


// -------------------Symmetrized Neural layer-----------------------------
SymmNeuralLayer::SymmNeuralLayer(const int& units, const std::string& activation, 
    const int& input_dim)
  : NeuralLayer(units, activation, input_dim)
{
  num_params_ = kernel_.rows()+bias_.size();
}

void SymmNeuralLayer::init_parameters(random_engine& rng, const double& sigma) 
{
  std::normal_distribution<double> random_normal(1.0,sigma);
  for (int i=0; i<kernel_.rows(); ++i) {
    double w = random_normal(rng);
    for (int j=0; j<kernel_.cols(); ++j) {
      kernel_(i,j) = w;
    }
  }
  for (int i=0; i<bias_.size(); ++i) {
    bias_(i) = random_normal(rng);
  }
}

const double& SymmNeuralLayer::get_parameter(const int& id) const
{
  if (id < kernel_.rows()) {
    return kernel_(id,0); 
  }
  else if (id < num_params_) {
    int n = id-kernel_.rows();
    return *(bias_.data()+n);
  }
  else {
    throw std::out_of_range("NeuralLayer::get_parameter: out-of-range 'id'");
  }
}
void SymmNeuralLayer::get_parameter_names(std::vector<std::string>& pnames, const int& pos) const
{
  std::string w = "w"+std::to_string(id_)+"_";
  int n = pos;
  for (int i=0; i<kernel_.rows(); ++i) {
      pnames[n] = w + std::to_string(i);
      n++;
  }
  std::string b = "b"+std::to_string(id_)+"_";
  for (int i=0; i<bias_.size(); ++i) {
    pnames[n++] = b + std::to_string(i);
  }
}
void SymmNeuralLayer::get_parameter_values(eig::real_vec& pvalues, const int& pos) const
{
  for (int i=0; i<kernel_.rows(); ++i)
    pvalues(pos+i) = kernel_(i,0);
  int n = pos+kernel_.rows();
  for (int i=0; i<bias_.size(); ++i)
    pvalues(n+i) = *(bias_.data()+i);
}

void SymmNeuralLayer::get_parameters(Vector& pvec, const int& start_pos) const
{
  for (int i=0; i<kernel_.rows(); ++i)
    pvec(start_pos+i) = kernel_(i,0);
  int n = start_pos+kernel_.rows();
  for (int i=0; i<bias_.size(); ++i)
    pvec(n+i) = *(bias_.data()+i);
}

void SymmNeuralLayer::update_parameters(const Vector& pvec, const int& start_pos)
{
  for (int i=0; i<kernel_.rows(); ++i){
    double w= pvec(start_pos+i);
      for(int j=0;j<kernel_.cols();++j){
        kernel_(i,j)=w;
      }
  
  }
  //std::cout << "kernel =\n" << kernel_ << "\n\n\n"; getchar();
  int n = start_pos+kernel_.rows();
  for (int i=0; i<bias_.size(); ++i)
    *(bias_.data()+i) = pvec(n+i);
}

void SymmNeuralLayer::update_parameter(const int& id, const double& value)
{
  if (id < kernel_.rows()) {
    kernel_(id,0)= value;
  }
  else if (id < num_params_) {
    int n = id-kernel_.rows();
    *(bias_.data()+n) = value;
  }
  else {
    throw std::out_of_range("NeuralLayer::get_parameter: out-of-range 'id'");
  }
}

int SymmNeuralLayer::derivative(Matrix& derivative, const int& num_total_params) const
{
  if (inlayer_ == nullptr) return 0;
  //std::cout<<"call this function-> derivative"<<std::endl;
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
    for (int m=0; m<kernel_.rows(); ++m) {
      if (m == q) {
        derivative(p++,q) = der_activation_(m)*inlayer_->output().sum();
      }
      else {
        derivative(p++,q) = 0.0;
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

int SymmNeuralLayer::back_propagate(const int& pid_end, const RowVector& backflow,
  Matrix& derivative, const int& use_col) const
{
  if (inlayer_ == nullptr) return 0;
  //std::cout<<"call this function-> back_propagate"<<std::endl;
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
  for (int m=0; m<kernel_.rows(); ++m) {
    derivative(p++,use_col) = der_activation_(m)*inlayer_->output().sum();
  }
  // derivaive wrt bias parameters
  for (int m=0; m<bias_.size(); ++m) {
    derivative(p++,use_col) = der_activation_(m);
  }
  // new backflow for the inner layer
  der_backflow_ = der_activation_.transpose() * kernel_;
  return inlayer_->back_propagate(pid_begin, der_backflow_, derivative, use_col);
}



} // end namespace ann
