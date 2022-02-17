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

namespace vmc {

enum class run_mode {normal, energy_function, sr_function};

class VMC //: public optimizer::Problem
{
public:
  VMC(const input::Parameters& inputs); 
  virtual ~VMC() {}
  int start(const input::Parameters& inputs, const run_mode& mode=run_mode::normal, 
    const bool& silent=false);
  void seed_rng(const int& seed=1);
  int run_simulation(const int& sample_size=-1, 
    const std::vector<int>& bc_list={-1});
  int run_simulation(const Eigen::VectorXd& varp);
  int reset_observables(void);
  int do_warmup_run(void); 
  int do_measure_run(const int& num_samples); 
  int energy_function(const Eigen::VectorXd& varp, double& en_mean, double& en_stddev,
    Eigen::VectorXd& grad);
  int operator()(const Eigen::VectorXd& varp, double& en_mean, double& en_stddev,
     Eigen::VectorXd& grad) { return energy_function(varp,en_mean,en_stddev,grad); }
  int build_config(const Eigen::VectorXd& varp, const bool& with_psi_grad); 
  int sr_function(const Eigen::VectorXd& vparms, double& en_mean, double& en_stddev,
    Eigen::VectorXd& grad, Eigen::MatrixXd& sr_matrix, 
    const int& sample_size=-1, const int& rng_seed=-1);
  //void get_vparm_values(var::parm_vector& varparms) 
  //  { varparms = config.vparm_values(); }
  void save_parameters(const var::parm_vector& parms) 
    { config.save_parameters(parms); }
  const int& num_measure_steps(void) const { return num_measure_steps_; } 
  const int& num_varp(void) const { return config.num_varparms(); } 
  const var::parm_vector& varp_values(void) { return config.vparm_values(); }
  const var::parm_vector& varp_lbound(void) const { return config.vparm_lbound(); }
  const var::parm_vector& varp_ubound(void) const { return config.vparm_ubound(); }
  const std::vector<std::string>& varp_names(void) const { return config.varp_names(); }
  RandomGenerator& rng(void) const { return config.rng(); }
  const double& hole_doping(void) const { return config.hole_doping(); }
  const std::vector<std::string>& xvar_names(void) const { return xvar_names_; }
  const std::vector<double>& xvar_values(void) const { return xvar_values_; }
  const ObservableSet& observable(void) const { return observables; }
  void finalize_results(void) { observables.finalize(); } 
  void print_results(void); 
  const std::string prefix_dir(void) const { return prefix_; }
  std::ostream& print_info(std::ostream& os) const { return os << info_str_.str(); }
  static void copyright_msg(std::ostream& os);

  void MPI_send_results(const mpi::mpi_communicator& mpi_comm, const mpi::proc& proc, 
    const int& msg_tag) { observables.MPI_send_results(mpi_comm, proc, msg_tag); }
  void MPI_recv_results(const mpi::mpi_communicator& mpi_comm, const mpi::proc& proc, 
    const int& msg_tag) { observables.MPI_recv_results(mpi_comm, proc, msg_tag); }
private:
  run_mode run_mode_{run_mode::normal};
  lattice::LatticeGraph graph;
  model::Hamiltonian model;
  SysConfig config;
  int rng_seed_;
  int num_sites_;
  int num_varparms_;
  // default BC_TWIST list
  std::vector<int> bc_default_{0};

  // observables
  ObservableSet observables;
  std::string prefix_{"./"};
  std::vector<std::string> xvar_names_;
  std::vector<double> xvar_values_;


  // mc parameters
  enum move_t {uphop, dnhop, exch, end};
  int num_measure_steps_{0}; 
  int num_warmup_steps_{0};
  int min_interval_{0};
  int max_interval_{0};
  int check_interval_{0};
  bool silent_mode_{false};

  mutable std::ostringstream info_str_;

  void make_info_str(const input::Parameters& inputs);
  void print_progress(const int& num_measurement, const int& num_measure_steps) const;
};

} // end namespace vmc

#endif
