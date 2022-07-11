/*---------------------------------------------------------------------------
* Copyright (C) 2015-2016 by Amal Medhi <amedhi@iisertvm.ac.in>.
* All rights reserved.
* Author: Amal Medhi
*----------------------------------------------------------------------------*/
#include "observables.h"
#include <boost/algorithm/string.hpp>

namespace vmc {

ObservableSet::ObservableSet() 
  : energy_("Energy")
  , energy_grad_("EnergyGradient")
  , sc_corr_("SC_Correlation")
  , sr_matrix_("SR_Matrix")
  , site_occupancy_("SiteOccupancy")
  , k_occupancy_("MomentumOccupancy")
{
}

//void ObservableSet::init(const input::Parameters& inputs, 
//    void (&print_copyright)(std::ostream& os), const lattice::LatticeGraph& graph, 
//    const model::Hamiltonian& model, const SysConfig& config)
void ObservableSet::init(const input::Parameters& inputs, 
  const lattice::LatticeGraph& graph, const model::Hamiltonian& model, 
  const SysConfig& config, const std::string& prefix)
{
  // file open mode
  std::string mode = inputs.set_value("mode", "NEW");
  boost::to_upper(mode);
  if (mode=="APPEND") replace_mode_ = false;
  else replace_mode_ = true;
  // check which observables to calculate
  //for (auto& obs : *this) obs.get().check_on(inputs, replace_mode_);
  // heading message
  //print_copyright(headstream_);
  //model.print_info(headstream_);
  num_xvars_ = 0; 

  // files
  energy_.set_ofstream(prefix);
  energy_grad_.set_ofstream(prefix);
  sc_corr_.set_ofstream(prefix);
  sr_matrix_.set_ofstream(prefix);
  site_occupancy_.set_ofstream(prefix);
  k_occupancy_.set_ofstream(prefix);

  // switch on required observables
  energy_.check_on(inputs, replace_mode_);
  energy_grad_.check_on(inputs, replace_mode_);
  if (energy_grad_) energy_.switch_on();
  sc_corr_.check_on(inputs,replace_mode_);
  sr_matrix_.check_on(inputs,replace_mode_);
  site_occupancy_.check_on(inputs,replace_mode_);
  k_occupancy_.check_on(inputs,replace_mode_);

  // set up observables
  if (energy_) energy_.setup(graph,model);
  if (energy_grad_) energy_grad_.setup(config);
  if (sc_corr_) sc_corr_.setup(graph);
  if (sr_matrix_) sr_matrix_.setup(graph,config);
  if (site_occupancy_) site_occupancy_.setup(graph,config);
  if (k_occupancy_) k_occupancy_.setup(graph,config);
}

void ObservableSet::reset(void)
{
  if (energy_) energy_.reset();
  if (energy_grad_) energy_grad_.reset();
  if (sc_corr_) sc_corr_.reset();
  if (sr_matrix_) sr_matrix_.reset();
  if (site_occupancy_) site_occupancy_.reset();
  if (k_occupancy_) k_occupancy_.reset();
}

void ObservableSet::reset_grand_data(void)
{
  if (energy_) energy_.reset_grand_data();
  if (energy_grad_) energy_grad_.reset_grand_data();
  if (sc_corr_) sc_corr_.reset_grand_data();
  if (sr_matrix_) sr_matrix_.reset_grand_data();
  if (site_occupancy_) site_occupancy_.reset_grand_data();
  if (k_occupancy_) k_occupancy_.reset_grand_data();
}

void ObservableSet::save_results(void)
{
  if (energy_) energy_.save_result();
  if (energy_grad_) energy_grad_.save_result();
  if (sc_corr_) sc_corr_.save_result();
  if (sr_matrix_) sr_matrix_.save_result();
  if (site_occupancy_) site_occupancy_.save_result();
  if (k_occupancy_) k_occupancy_.save_result();
}

void ObservableSet::avg_grand_data(void)
{
  if (energy_) energy_.avg_grand_data();
  if (energy_grad_) energy_grad_.avg_grand_data();
  if (sc_corr_) sc_corr_.avg_grand_data();
  if (sr_matrix_) sr_matrix_.avg_grand_data();
  if (site_occupancy_) site_occupancy_.avg_grand_data();
  if (k_occupancy_) k_occupancy_.avg_grand_data();
}

int ObservableSet::do_measurement(const lattice::LatticeGraph& graph, 
    const model::Hamiltonian& model, const SysConfig& config)
{
  if (energy_) energy_.measure(graph,model,config);
  if (energy_grad_) {
    if (!energy_) 
      throw std::logic_error("ObservableSet::measure: dependency not met for 'energy'");
    energy_grad_.measure(config, energy_.config_value().sum());
  }
  if (sc_corr_) sc_corr_.measure(graph,model,config);
  if (sr_matrix_) {
    if (!energy_grad_) 
      throw std::logic_error("ObservableSet::measure: dependency not met for 'sr_matrix_'");
    sr_matrix_.measure(energy_grad_.grad_logpsi());
  }
  if (site_occupancy_) site_occupancy_.measure(graph, config);
  if (k_occupancy_) k_occupancy_.measure(graph, config);
  return 0;
}

void ObservableSet::finalize(void)
{
  if (energy_grad_) {
    energy_grad_.finalize();
  }
  if (k_occupancy_) {
    k_occupancy_.finalize();
  }
}

void ObservableSet::as_functions_of(const std::vector<std::string>& xvars)
{
  xvars_ = xvars;
  num_xvars_ = xvars_.size();
}

void ObservableSet::as_functions_of(const std::string& xvar)
{
  xvars_ = {xvar};
  num_xvars_ = 1;
}

void ObservableSet::switch_off(void) {
  energy_.switch_off();
  energy_grad_.switch_off();
  sc_corr_.switch_off();
  sr_matrix_.switch_off();
  site_occupancy_.switch_off();
  k_occupancy_.switch_off();
}

void ObservableSet::print_heading(void)
{
  energy_.print_heading(headstream_.rdbuf()->str(),xvars_);
  energy_grad_.print_heading(headstream_.rdbuf()->str(),xvars_);
  sc_corr_.print_heading(headstream_.rdbuf()->str(),xvars_);
  sr_matrix_.print_heading(headstream_.rdbuf()->str(),xvars_);
  site_occupancy_.print_heading(headstream_.rdbuf()->str(),xvars_);
  k_occupancy_.print_heading(headstream_.rdbuf()->str(),xvars_);
}

void ObservableSet::print_results(const std::vector<double>& xvals) 
{
  if (num_xvars_ != xvals.size()) 
    throw std::invalid_argument("Observables::print_result: 'x-vars' size mismatch");
  if (energy_) {
    energy_.print_heading(headstream_.rdbuf()->str(),xvars_);
    energy_.print_result(xvals);
  }
  if (energy_grad_) {
    energy_grad_.print_heading(headstream_.rdbuf()->str(),xvars_);
    energy_grad_.print_result(xvals);
  }
  if (sc_corr_) {
    sc_corr_.print_heading(headstream_.rdbuf()->str(),xvars_);
    sc_corr_.print_result(xvals);
  }
  if (site_occupancy_) {
    site_occupancy_.print_heading(headstream_.rdbuf()->str(),xvars_);
    site_occupancy_.print_result(xvals);
  }
  if (k_occupancy_) {
    k_occupancy_.print_heading(headstream_.rdbuf()->str(),xvars_);
    k_occupancy_.print_result(xvals);
  }
}

void ObservableSet::print_results(const double& xval) 
{
  if (num_xvars_ != 1) 
    throw std::invalid_argument("ObservableSet::print_result: 'x-vars' size mismatch");
  std::vector<double> xvals{xval};
  if (energy_) {
    energy_.print_heading(headstream_.rdbuf()->str(),xvars_);
    energy_.print_result(xvals);
  }
  if (energy_grad_) {
    energy_grad_.print_heading(headstream_.rdbuf()->str(),xvars_);
    energy_grad_.print_result(xvals);
  }
  if (sc_corr_) {
    sc_corr_.print_heading(headstream_.rdbuf()->str(),xvars_);
    sc_corr_.print_result(xvals);
  }
  if (site_occupancy_) {
    site_occupancy_.print_heading(headstream_.rdbuf()->str(),xvars_);
    site_occupancy_.print_result(xvals);
  }
  if (k_occupancy_) {
    k_occupancy_.print_heading(headstream_.rdbuf()->str(),xvars_);
    k_occupancy_.print_result(xvals);
  }
}

void ObservableSet::MPI_send_results(const mpi::mpi_communicator& mpi_comm, 
  const mpi::proc& proc, const int& msg_tag)
{
  if (energy_) energy_.MPI_send_data(mpi_comm, proc, msg_tag);
  if (energy_grad_) energy_grad_.MPI_send_data(mpi_comm, proc, msg_tag);
  if (sc_corr_) sc_corr_.MPI_send_data(mpi_comm, proc, msg_tag);
  if (sr_matrix_) sr_matrix_.MPI_send_data(mpi_comm, proc, msg_tag);
  if (site_occupancy_) site_occupancy_.MPI_send_data(mpi_comm, proc, msg_tag);
  if (k_occupancy_) k_occupancy_.MPI_send_data(mpi_comm, proc, msg_tag);
}

void ObservableSet::MPI_recv_results(const mpi::mpi_communicator& mpi_comm, 
  const mpi::proc& proc, const int& msg_tag)
{
  if (energy_) energy_.MPI_add_data(mpi_comm, proc, msg_tag);
  if (energy_grad_) energy_grad_.MPI_add_data(mpi_comm, proc, msg_tag);
  if (sc_corr_) sc_corr_.MPI_add_data(mpi_comm, proc, msg_tag);
  if (sr_matrix_) sr_matrix_.MPI_add_data(mpi_comm, proc, msg_tag);
  if (site_occupancy_) site_occupancy_.MPI_add_data(mpi_comm, proc, msg_tag);
  if (k_occupancy_) k_occupancy_.MPI_add_data(mpi_comm, proc, msg_tag);
}

} // end namespace vmc

