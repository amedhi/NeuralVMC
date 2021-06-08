/*
* @Author: Amal Medhi, amedhi@mbpro
* @Date:   2020-05-27 15:52:49
* @Last Modified by:   Amal Medhi, amedhi@mbpro
* @Last Modified time: 2020-06-02 16:21:00
*/
#include <locale>
#include "sign_layer.h"

namespace ann {

SignLayer::SignLayer(const std::string& activation, const int& input_dim)
  : NeuralLayer(1, activation, input_dim)
{
  cos_thetak_.resize(input_dim);
  sin_thetak_.resize(input_dim);
}

void SignLayer::set_output_layer(NeuralLayer* layer) 
{
  throw std::out_of_range("SignLayer::set_output_layer: can't take output layer");
}

int SignLayer::update_forward(const Vector& input) 
{
	re_phase_ = 0.0;
	im_phase_ = 0.0;
  for (int k=0; k<input_dim_; ++k) {
    double sum = 0.0;
  	for (int j=0; j<k; ++j) {
  		sum += kernel_(0,k-j)*input(j);
  	}
  	int i = 1;
  	for (int j=k+1; j<input_dim_; ++j) {
  		sum += kernel_(0,input_dim_-i)*input(j);
  		i++;
  	}
  	double theta = sum + bias_(0);
  	cos_thetak_[k] = std::cos(theta);
  	sin_thetak_[k] = std::sin(theta);
  	re_phase_ += cos_thetak_[k];
  	im_phase_ += sin_thetak_[k];
  }
  lin_output_[0] = std::arg(std::complex<double>(re_phase_,im_phase_));
  output_ = lin_output_;
  return 0;
}

int SignLayer::update_forward(const Vector& new_input, 
  const std::vector<int>& new_elems, const Vector& input_changes) 
{
	update_forward(new_input);
  return 0;
}

int SignLayer::feed_forward(const Vector& input) const
{
  /* Feed forward without changing the state of the layer */
	double Re = 0.0;
	double Im = 0.0;
  for (int k=0; k<input_dim_; ++k) {
    double sum = 0.0;
  	for (int j=0; j<k; ++j) {
  		sum += kernel_(0,k-j)*input(j);
  	}
  	int i = 1;
  	for (int j=k+1; j<input_dim_; ++j) {
  		sum += kernel_(0,input_dim_-i)*input(j);
  		i++;
  	}
  	double theta = sum + bias_(0);
  	Re += std::cos(theta);
  	Im += std::sin(theta);
  }
  lin_output_tmp_[0] = std::arg(std::complex<double>(Re,Im));
  output_tmp_ = lin_output_tmp_;
  return 0;
}

int SignLayer::feed_forward(const Vector& new_input, 
  const std::vector<int>& new_elems, const Vector& input_changes) const
{
  /* Feed forward without changing the state of the layer */
  feed_forward(new_input);
  return 0;
}

Vector SignLayer::get_new_output(const Vector& input) const
{
  /* Give output without changing the state of the layer */
	double Re = 0.0;
	double Im = 0.0;
  for (int k=0; k<input_dim_; ++k) {
    double sum = 0.0;
  	for (int j=0; j<k; ++j) {
  		sum += kernel_(0,k-j)*input(j);
  	}
  	int i = 1;
  	for (int j=k+1; j<input_dim_; ++j) {
  		sum += kernel_(0,input_dim_-i)*input(j);
  		i++;
  	}
  	double theta = sum + bias_(0);
  	Re += std::cos(theta);
  	Im += std::sin(theta);
  }
  lin_output_tmp_[0] = std::arg(std::complex<double>(Re,Im));
  return lin_output_tmp_;
}

int SignLayer::derivative(Matrix& derivative, const int& num_total_params) const
{
  // derivative of the SINGLE output unit
  int pid_begin = num_total_params-num_params_;
  // derivaive wrt weight parameters in this layer
  /* since parameters are COUNTED COLUMN wise */
  int p = pid_begin;
  for (int n=0; n<kernel_.cols(); ++n) {
    double sum1 = 0.0;
    double sum2 = 0.0;
    for (int k=0; k<n; ++k) {
    	int s = inlayer_->output()(input_dim_+k-n);
    	sum1 += cos_thetak_[k]*s;
    	sum2 += sin_thetak_[k]*s;
    }
    for (int k=n; k<input_dim_; ++k) {
    	int s = inlayer_->output()(k-n);
    	sum1 += cos_thetak_[k]*s;
    	sum2 += sin_thetak_[k]*s;
    }
    derivative(p++,0) = (re_phase_*sum1+im_phase_*sum2)
                       /(re_phase_*re_phase_ + im_phase_*im_phase_);
  }
  // derivaive wrt the bias parameter in this layer
  derivative(p++,0) = 1.0;

  //std::cout << "derivative = " << derivative << "\n";
  return 0;
}



}