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
#include "dtype.h"

namespace ann {

class NeuralLayer 
{
public:
  NeuralLayer(const int& units, const std::string& activation="None", 
    const int& input_dim=1);
  virtual ~NeuralLayer() {}
  void set_id(const int& id) { id_=id; }
  void set_input(const RealVector& v) { input_=v; }
  void set_input(const IntVector& v) { input_=v.cast<double>(); }
  void set_kernel(const RealMatrix& w) { kernel_=w; }
  void set_bias(const RealVector& b) { bias_=b; }
  const int& num_units(void) const { return num_units_; }
  const int& input_dim(void) const { return input_dim_; }
  const int& num_params(void) const { return num_params_; } 
  virtual void set_input_layer(NeuralLayer* layer) { inlayer_=layer; }
  virtual void set_output_layer(NeuralLayer* layer) { outlayer_=layer; }
  virtual void init_parameters(random_engine& rng, const double& sigma); 
  virtual void save_parameters(std::ofstream& fout) const; 
  virtual void load_parameters(std::ifstream& fin); 
  virtual const double& get_parameter(const int& id) const;
  virtual void get_parameters(RealVector& pvec, const int& start_pos) const;
  virtual void get_parameter_names(std::vector<std::string>& pnames, const int& pos) const;
  virtual void get_parameter_values(RealVector& pvalues, const int& pos) const;
  virtual void update_parameters(const RealVector& pvec, const int& start_pos);
  virtual void update_parameter(const int& id, const double& value);
  virtual int update_forward(const RealVector& input);
  virtual int update_forward(const RealVector& new_input, const std::vector<int>& new_elems, 
    const RealVector& input_changes); 
  virtual int feed_forward(const RealVector& input) const; 
  virtual int feed_forward(const RealVector& new_input, const std::vector<int>& new_elems, const RealVector& input_changes) const;
  const RealVector& output(void) const { return output_; } 
  const RealVector& new_output(void) const { return output_tmp_; }
  virtual RealVector get_new_output(const RealVector& input) const; 
  const RealVector& linear_output(void) const { return lin_output_; }
  virtual int derivative(RealMatrix& derivative, const int& num_total_params) const;
  RealVector derivative_fwd(const int& lid, const int& pid) const; 
  virtual int back_propagate(const int& pid_end, const RealRowVector& backflow,
    RealMatrix& derivative, const int& use_col) const;
protected:
  int id_{0};
  int num_units_{1};
  int input_dim_{1};
  int num_params_{1};
  RealVector input_;
  RealMatrix kernel_;
  RealVector bias_;
  RealVector output_;
  RealVector lin_output_;
  mutable RealVector output_changes_;
  mutable RealVector output_tmp_;
  mutable RealVector lin_output_tmp_;
  mutable RealVector der_activation_;
  mutable RealRowVector der_backflow_;
  mutable RealVector derivative_;
  std::shared_ptr<Activation> activation_{nullptr};
  NeuralLayer* inlayer_{nullptr};
  NeuralLayer* outlayer_{nullptr};
};


} // end namespace ann


#endif
