/*----------------------------------------------------------------------------
* @Author: Amal Medhi
* @Date:   2023-12-25 20:17:54
* @Last Modified by:   amedhi
* @Last Modified time: 2023-12-25 23:26:32
*----------------------------------------------------------------------------*/
#ifndef RBM_H
#define RBM_H

#include <vector>
#include <string>
#include <memory>
#include <tuple>
#include "abstract_net.h"
#include "neural_layer.h"
#include "../lattice/lattice.h"

namespace ann {

class RBM: public AbstractNet
{
public:
  RBM();
  RBM(const lattice::Lattice& lattice, const input::Parameters& inputs);
  ~RBM() { for (auto& p : layers_) delete p; layers_.clear(); }
  int construct(const lattice::Lattice& lattice, const input::Parameters& inputs);
  virtual int add_layer(const int& units, const std::string& activation="None", 
    const int& input_dim=0) override;
  virtual int add_sign_layer(const int& input_dim) override;
  int compile(void) override { return 0; };
  void init_parameter_file(const std::string& prefix) override;
  void init_parameters(random_engine& rng, const double& sigma) override;
  void save_parameters(void) const override;
  void load_parameters(const std::string& load_path) override;
  const int& num_params(void) const override { return num_params_; }
  const int& num_output_units(void) const override { return num_output_units_; }
  //void set_input(const Vector& input) { front().set_input(input); }
  const double& get_parameter(const int& id) const;
  void get_parameters(Vector& pvec) const;
  void get_parameter_names(std::vector<std::string>& pnames, 
    const int& pos=0) const override;
  void get_parameter_values(RealVector& pvalues, const int& pos=0) const override;
  void get_parameter_vector(std::vector<double>& pvalues, const int& pos) const override;
  void update_parameters(const RealVector& pvec, const int& pos=0) override;
  void update_parameter(const int& id, const double& value) override;
  void do_update_run(const Vector& input) override; 
  void do_update_run(const Vector& new_input, const std::vector<int> new_elems) override; 
  //void run(const eig::real_vec& input); 
  const Vector& output(void) const override { return output_; }
  Vector get_new_output(const Vector& input) const override;
  Vector get_new_output(const Vector& new_input, const std::vector<int> new_elems) const override; 
  void get_gradient(Vector& grad, const int& pos) const override;
  void get_log_gradient(Vector& grad, const int& pos) const override;
  void get_gradient(Matrix& grad_mat) const override;
  void get_log_gradient(Matrix& grad_mat) const override;
protected:
  int num_layers_{0};
  int num_output_units_{1}; // always 1, not same as 'num_hidden_units'

  // implementation of translational symmetry
  int num_sites_{0};
  int num_basis_sites_{1};
  bool symmetry_{true};
  int num_tsymms_{1};
  IntMatrix tsymm_map_;

  // structure
  int num_visible_units_{1};
  int num_hidden_units_{1};
  int num_hblocks_{1};
  int num_kernel_params_{0};
  int num_hbias_params_{0};
  int num_params_{0};
  RealVector vbias_;
  RealVector hbias_;
  RealMatrix kernel_;
  RealVector input_;
  RealVector lin_output_;
  RealVector output_;
  RealVector tmp_output_;
  mutable RealVector tanh_output_;

  // parameters
  RealVector pvector_; // all parameters
  using idx_list = std::vector<std::pair<int,int>>;
  std::vector<idx_list> kernel_params_map_; // locations where a paramater appear
  std::vector<std::vector<int>> bias_params_map_; // locations where a paramater appear

  //std::vector<NeuralLayer> layers_;
  std::vector<NeuralLayer*> layers_;
  std::vector<int> num_params_fwd_;
  //Vector output_;
  mutable Vector input_changes_;
  mutable Matrix gradient_;
  mutable Matrix log_gradient_;
  // parameter file
  std::string prefix_{""};

  int update_kernel_params(const RealVector& params);
  int update_hbias_params(const RealVector& params);
  Matrix row_translate(const RealMatrix& mat, const int& T) const;
};


} // end namespace ann
#endif
