/*---------------------------------------------------------------------------
* @Author: Amal Medhi
* @Date:   2018-12-29 10:48:33
* @Last Modified by:   amedhi
* @Last Modified time: 2019-01-08 23:27:00
*----------------------------------------------------------------------------*/
#ifndef NEURAL_LAYER_H
#define NEURAL_LAYER_H

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <Eigen/Core>
#include "activation.h"

namespace ann {

class NeuralLayer 
{
public:
  NeuralLayer(const int& units, const std::string& activation="None", 
    const int& input_dim=1);
  ~NeuralLayer() {}
  void set_id(const int& id) { id_=id; }
  void set_input(const Vector& v) { input_=v; }
  void set_input(const ivector& v) { input_=v.cast<double>(); }
  void set_kernel(const Matrix& w) { kernel_=w; }
  void set_bias(const Vector& b) { bias_=b; }
  void set_input_layer(NeuralLayer* layer) { inlayer_=layer; }
  void set_output_layer(NeuralLayer* layer) { outlayer_=layer; }
  const int& num_units(void) const { return num_units_; }
  const int& num_params(void) const { return num_params_; } 
  void init_parameters(random_engine& rng, const double& sigma); 
  const double& get_parameter(const int& id) const;
  void get_parameters(Vector& pvec, const int& start_pos) const;
  void get_parameter_names(std::vector<std::string>& pnames, const int& pos) const;
  void get_parameter_values(eig::real_vec& pvalues, const int& pos) const;
  void update_parameters(const Vector& pvec, const int& start_pos);
  void update_parameter(const int& id, const double& value);
  int update_forward(const Vector& input);
  int update_forward(const Vector& new_input, const std::vector<int>& new_elems, 
    const Vector& input_changes); 
  const Vector& output(void) const { return output_; }
  const Vector& new_output(void) const { return output_tmp_; }
  int feed_forward(const Vector& input) const; 
  int feed_forward(const Vector& new_input, 
  const std::vector<int>& new_elems, const Vector& input_changes) const;
  Vector get_new_output(const Vector& input) const; 
  const Vector& linear_output(void) const { return lin_output_; }
  int derivative(Matrix& derivative, const int& num_total_params) const;
  Vector derivative_fwd(const int& lid, const int& pid) const; 
private:
  int id_{0};
  int num_units_{1};
  int input_dim_{1};
  int num_params_{1};
  Vector input_;
  Matrix kernel_;
  Vector bias_;
  Vector output_;
  Vector lin_output_;
  mutable Vector output_changes_;
  mutable Vector output_tmp_;
  mutable Vector lin_output_tmp_;
  mutable Vector der_activation_;
  mutable RowVector der_backflow_;
  mutable Vector derivative_;
  std::shared_ptr<Activation> activation_{nullptr};
  NeuralLayer* inlayer_{nullptr};
  NeuralLayer* outlayer_{nullptr};

  int back_propagate(const int& pid_end, const RowVector& backflow,
    Matrix& derivative, const int& use_col) const;
};

/*
class AbstractLayer
{
public:
  virtual ~AbstractLayer() {}
  virtual void set_kernel(const Matrix& w) = 0; 
  virtual void set_bias(const Vector& b) = 0;
  virtual void set_input(const Vector& v) { input_=v; }
  virtual void set_input_layer(AbstractLayer* inlayer) = 0;
  virtual void set_output_layer(AbstractLayer* outlayer) = 0;
  virtual const Matrix& get_kernel(void) const { return kernel_; } 
  virtual const Vector& get_bias(void) const { return bias_; }
  virtual const Vector& get_input(void) const { return input_; }
  virtual const int& num_params(void) const = 0; 
  virtual const double& get_parameter(const int& id) const = 0;
  virtual void update_parameter(const int& id, const double& value) = 0;
  virtual Vector output(void) = 0; 
  virtual Vector derivative(const int& id) const = 0; 
  virtual Vector get_output(void) const = 0; 
  virtual const int& num_units(void) const { return num_units_; }
protected:
  static int num_layers_;
  int id_{0};
  int num_units_{1};
  int num_params_{1};
  Vector input_;
  Vector output_;
  Matrix kernel_;
  Vector bias_;
};

class InputLayer : public AbstractLayer
{
public:
  InputLayer(const int& units=1); 
  ~InputLayer() { num_layers_--; }
  void set_kernel(const Matrix& w) override {}
  void set_bias(const Vector& b) override {}
  void set_input_layer(AbstractLayer* layer) override {}
  void set_output_layer(AbstractLayer* layer) override { outlayer_=layer; }
  const int& num_params(void) const override { return num_params_; } 
  const double& get_parameter(const int& id) const override { return zero_; }
  void update_parameter(const int& id, const double& value) override {}
  Vector get_output(void) const override { return input_; } 
  Vector output(void) override { return input_; } 
  Vector derivative(const int& id) const override 
    { return Vector::Zero(input_.size()); }
private:
  AbstractLayer* outlayer_{nullptr};
  double zero_{0.0};
};
class Layer
{
public:
  //Layer() { id_=num_layers_++; }
  Layer(const int& units, const std::string& activation="None", 
    const int& input_dim=1);
  //Layer(const Layer& layer);
  ~Layer() { --num_layers_; } // inlayer_.reset(nullptr); outlayer_.reset(nullptr); }
  void set_input_dim(const int& input_dim) 
  { 
    input_dim_=input_dim; 
    kernel_=Matrix::Ones(num_units_,input_dim_);
    input_=Vector::Zero(input_dim_);
  }
  void set_input(const Vector& input) { input_=input; } 
  void set_input_layer(Layer* inlayer);
  void set_output_layer(Layer* outlayer);
  void set_kernel(const Matrix& w) { kernel_=w; }
  void set_bias(const Vector& b) { bias_=b; }
  const int& num_units(void) const { return num_units_; }
  Vector get_output(void) const; 
  const Vector& get_input(void) const { return input_; } 
  std::string name_;
  Layer* inlayer_{nullptr};
  Layer* outlayer_{nullptr};
private:
  static int num_layers_;
  int id_{0};
  int num_units_{1};
  int input_dim_{1};
  std::shared_ptr<Activation> activation_{nullptr};
  Matrix kernel_;
  Vector bias_;
  Vector input_;
};
*/




} // end namespace ann


#endif