/*---------------------------------------------------------------------------
* Author: Amal Medhi
* Date:   2017-02-12 13:19:36
* Last Modified by:   Amal Medhi, amedhi@macbook
* Last Modified time: 2017-05-20 11:15:18
* Copyright (C) Amal Medhi, amedhi@iisertvm.ac.in
*----------------------------------------------------------------------------*/
#ifndef VMC_H
#define VMC_H

#include "../scheduler/worker.h"
#include "../scheduler/mpi_comm.h"
#include "../lattice/lattice.h"
#include "../lattice/graph.h"
#include "../model/model.h"
#include "./observables.h"
#include "./sysconfig.h"
//#include "../optimizer/optimizer.h"

namespace vmc {

enum class run_mode {normal, energy_function, sr_function};

class VMC //: public optimizer::Problem
{
public:
  VMC(const input::Parameters& inputs); 
  virtual ~VMC() {}
  int init(const input::Parameters& inputs, const run_mode& mode=run_mode::normal, 
    const bool& silent=false);
  int do_warmup(const int& num_steps=-1);
  int do_steps(const int& num_steps=1);
  int run_simulation(const int& sample_size=-1);
  int run_simulation(const Eigen::VectorXd& varp);
  void save_parameters(void) { config.save_parameters(); }
  bool not_done(void) const;
  double energy_function(const Eigen::VectorXd& varp, Eigen::VectorXd& grad);
  double operator()(const Eigen::VectorXd& varp, Eigen::VectorXd& grad) 
    { return energy_function(varp, grad); }
  //double sr_function(const Eigen::VectorXd& vparms, Eigen::VectorXd& grad, 
  //  Eigen::MatrixXd& sr_matrix, const int& sample_size=-1);
  int sr_function(const Eigen::VectorXd& varp, double& en_mean, 
    double& en_stddev, Eigen::VectorXd& grad, Eigen::MatrixXd& sr_matrix, 
    const int& sample_size=-1, const int& rng_seed=-1);
  //void get_vparm_values(var::parm_vector& varparms) 
  //  { varparms = config.vparm_values(); }
  const int& num_varp(void) const { return config.num_varparms(); } 
  const var::parm_vector& varp_values(void) { return config.vparm_values(); }
  const var::parm_vector& varp_lbound(void) const { return config.vparm_lbound(); }
  const var::parm_vector& varp_ubound(void) const { return config.vparm_ubound(); }
  const std::vector<std::string>& varp_names(void) const { return config.varp_names(); }
  RandomGenerator& rng(void) const { return config.rng(); }
  const double& hole_doping(void) const { return config.hole_doping(); }
  const std::vector<std::string>& xvar_names(void) const { return xvar_names_; }
  const std::vector<double>& xvar_values(void) const { return xvar_values_; }
  const std::string prefix_dir(void) const { return prefix_; }
  void print_results(void); 
  std::ostream& print_info(std::ostream& os) const { return model.print_info(os); }
  static void copyright_msg(std::ostream& os);

private:
  run_mode run_mode_{run_mode::normal};
  lattice::LatticeGraph graph;
  model::Hamiltonian model;
  SysConfig config;
  int num_sites_;
  int num_varparms_;

  // observables
  ObservableSet observables;
  std::string prefix_{"./"};
  std::vector<std::string> xvar_names_;
  std::vector<double> xvar_values_;

  // mc parameters
  enum move_t {uphop, dnhop, exch, end};
  int samples_required_{0}; 
  int samples_collected_{0}; 
  //int num_measure_steps_{0}; 
  int num_warmup_steps_{0};
  int min_interval_{0};
  int max_interval_{0};
  int check_interval_{0};
  int skip_count_{0};
  bool silent_mode_{false};

  mutable std::ostringstream info_str_;

  void make_info_str(const input::Parameters& inputs);
  void print_progress(const int& num_measurement, const int& num_measure_steps) const;
};

} // end namespace vmc

#endif