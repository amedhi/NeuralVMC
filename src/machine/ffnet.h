/*
* @Author: Amal Medhi
* @Date:   2018-12-29 20:17:54
* @Last Modified by:   amedhi
* @Last Modified time: 2019-01-08 23:26:32
*----------------------------------------------------------------------------*/
#ifndef FFNET_H
#define FFNET_H

#include <vector>
#include <string>
#include <memory>
#include "abstract_net.h"
#include "neural_layer.h"
#include "sign_layer.h"

namespace ann {

class FFNet: public AbstractNet
{
public:
  FFNet();
  ~FFNet() { for (auto& p : layers_) delete p; layers_.clear(); }
  virtual int add_layer(const int& units, const std::string& activation="None", 
    const int& input_dim=0) override;
  virtual int add_sign_layer(const int& input_dim) override;
  int compile(void) override;
  void init_parameters(random_engine& rng, const double& sigma) override;
  const int& num_params(void) const override { return num_params_; }
  const int& num_output_units(void) const override { return layers_.back()->num_units(); }
  //void set_input(const Vector& input) { front().set_input(input); }
  const double& get_parameter(const int& id) const;
  void get_parameters(Vector& pvec) const;
  void get_parameter_names(std::vector<std::string>& pnames, 
    const int& pos=0) const override;
  void get_parameter_values(Vector& pvalues, const int& pos=0) const override;
  void get_parameter_vector(std::vector<double>& pvalues, const int& pos) const override;
  void update_parameters(const Vector& pvec, const int& pos=0) override;
  void update_parameter(const int& id, const double& value) override;
  void do_update_run(const Vector& input) override; 
  void do_update_run(const Vector& new_input, const std::vector<int> new_elems) override; 
  //void run(const eig::real_vec& input); 
  const Vector& output(void) const override { return layers_.back()->output(); }
  Vector get_new_output(const Vector& input) const override;
  Vector get_new_output(const Vector& new_input, const std::vector<int> new_elems) const override; 
  const Matrix& get_gradient(void) const override;
protected:
  int num_layers_{0};
  int num_params_{0};
  //std::vector<NeuralLayer> layers_;
  std::vector<NeuralLayer*> layers_;
  std::vector<int> num_params_fwd_;
  //Vector output_;
  mutable Vector input_changes_;
  mutable Matrix gradient_;
};

class SymmFFNet: public FFNet
{
public:
  SymmFFNet() : FFNet() {}
  ~SymmFFNet() {}
  virtual int add_layer(const int& units, const std::string& activation="None", 
    const int& input_dim=0) override;
  /*
  void init_parameters(random_engine& rng, const double& sigma) override;
  const double& get_parameter(const int& id) const;
  void get_parameters(Vector& pvec) const;
  void get_parameter_names(std::vector<std::string>& pnames, 
    const int& pos=0) const override;
  void get_parameter_values(Vector& pvalues, const int& pos=0) const override;
  void get_parameter_vector(std::vector<double>& pvalues, const int& pos) const override;
  void update_parameters(const Vector& pvec, const int& pos=0) override;
  void update_parameter(const int& id, const double& value) override;
  const Matrix& get_gradient(void) const override;
  */
private:
};

} // end namespace ann
#endif
