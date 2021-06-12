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
#include <fstream>
#include <iomanip>
#include <Eigen/Core>
#include "activation.h"

namespace ann {

class NeuralLayer 
{
public:
  NeuralLayer(const int& units, const std::string& activation="None", 
    const int& input_dim=1);
  virtual ~NeuralLayer() {}
  void set_id(const int& id) { id_=id; }
  void set_input(const Vector& v) { input_=v; }
  void set_input(const ivector& v) { input_=v.cast<double>(); }
  void set_kernel(const Matrix& w) { kernel_=w; }
  void set_bias(const Vector& b) { bias_=b; }
  const int& num_units(void) const { return num_units_; }
  const int& input_dim(void) const { return input_dim_; }
  const int& num_params(void) const { return num_params_; } 
  virtual void set_input_layer(NeuralLayer* layer) { inlayer_=layer; }
  virtual void set_output_layer(NeuralLayer* layer) { outlayer_=layer; }
  virtual void init_parameters(random_engine& rng, const double& sigma); 
  virtual void save_parameters(std::ofstream& fout) const; 
  virtual void load_parameters(std::ifstream& fin); 
  virtual const double& get_parameter(const int& id) const;
  virtual void get_parameters(Vector& pvec, const int& start_pos) const;
  virtual void get_parameter_names(std::vector<std::string>& pnames, const int& pos) const;
  virtual void get_parameter_values(eig::real_vec& pvalues, const int& pos) const;
  virtual void update_parameters(const Vector& pvec, const int& start_pos);
  virtual void update_parameter(const int& id, const double& value);
  virtual int update_forward(const Vector& input);
  virtual int update_forward(const Vector& new_input, const std::vector<int>& new_elems, 
    const Vector& input_changes); 
  virtual int feed_forward(const Vector& input) const; 
  virtual int feed_forward(const Vector& new_input, const std::vector<int>& new_elems, const Vector& input_changes) const;
  const Vector& output(void) const { return output_; } 
  const Vector& new_output(void) const { return output_tmp_; }
  virtual Vector get_new_output(const Vector& input) const; 
  const Vector& linear_output(void) const { return lin_output_; }
  virtual int derivative(Matrix& derivative, const int& num_total_params) const;
  Vector derivative_fwd(const int& lid, const int& pid) const; 
  virtual int back_propagate(const int& pid_end, const RowVector& backflow,
    Matrix& derivative, const int& use_col) const;
protected:
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
};

class SymmNeuralLayer : public NeuralLayer
{
public:
  SymmNeuralLayer(const int& units, const std::string& activation="None", 
    const int& input_dim=1);
  ~SymmNeuralLayer() {}
private:
   void init_parameters(random_engine& rng, const double& sigma) override; 
   const double& get_parameter(const int& id) const override;
  void get_parameter_names(std::vector<std::string>& pnames, const int& pos) const override;
  void get_parameter_values(eig::real_vec& pvalues, const int& pos) const override;
  void get_parameters(Vector& pvec, const int& start_pos) const override;
  void update_parameters(const Vector& pvec, const int& start_pos) override;
  void update_parameter(const int& id, const double& value) override;
  int derivative(Matrix& derivative, const int& num_total_params) const override;
  int back_propagate(const int& pid_end, const RowVector& backflow,
    Matrix& derivative, const int& use_col) const override;
};



} // end namespace ann


#endif
