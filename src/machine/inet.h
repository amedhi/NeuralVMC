/*----------------------------------------------------------------------------
* @Author: Amal Medhi
* @Date:   2023-12-25 20:17:54
* @Last Modified by:   amedhi
* @Last Modified time: 2023-12-25 23:26:32
*----------------------------------------------------------------------------*/
#ifndef INET_H
#define INET_H

#include <vector>
#include <string>
#include "abstract_net.h"
#include "neural_layer.h"
#include "../lattice/lattice.h"

namespace ann {

// Identity Network (Trival with output always 1)
class INet: public AbstractNet
{
public:
  INet();
  ~INet() {}
  virtual int add_layer(const int& units, const std::string& activation="None", 
    const int& input_dim=0) override;
  virtual int add_sign_layer(const int& input_dim) override;
  int compile(void) override { return 0; };
  void init_parameter_file(const std::string& save_path, const std::string& load_path) override;
  void init_parameters(random_engine& rng, const double& sigma) override;
  void save_parameters(void) const override;
  void load_parameters(void) override;
  const int& num_params(void) const override { return num_params_; }
  const int& num_output_units(void) const override { return num_output_units_; }
  //void set_input(const Vector& input) { front().set_input(input); }
  const double& get_parameter(const int& id) const;
  void get_parameters(RealVector& pvec) const;
  void get_parameter_names(std::vector<std::string>& pnames, 
    const int& pos=0) const override;
  void get_parameter_values(RealVector& pvalues, const int& pos=0) const override;
  void get_parameter_lbound(RealVector& lbound, const int& pos=0) const override;
  void get_parameter_ubound(RealVector& ubound, const int& pos=0) const override;
  void update_parameters(const RealVector& pvec, const int& pos=0) override;
  void update_parameter(const int& id, const double& value) override;
  void do_update_run(const RealVector& input) override; 
  void do_update_run(const RealVector& new_input, const std::vector<int> new_elems) override; 
  //void run(const eig::real_vec& input); 
  const RealVector& output(void) const override { return output_; }
  RealVector get_new_output(const RealVector& input) const override;
  RealVector get_new_output(const RealVector& new_input, const std::vector<int> new_elems) const override; 
  double get_new_output_ratio(const RealVector& input) const override { return 1.0; }
  double get_new_output_ratio(const RealVector& new_input, const std::vector<int> new_elems) const override 
    { return 1.0; }
  void get_gradient(RealMatrix& grad_mat) const override;
  void get_log_gradient(RealMatrix& grad_mat) const override;
protected:
  int num_layers_{0};
  int num_output_units_{1};
  int num_params_{0};
  RealVector output_;
};


} // end namespace ann
#endif
