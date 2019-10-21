/*---------------------------------------------------------------------------
* Author: Amal Medhi
* Date:   2017-03-09 15:19:43
* Last Modified by:   Amal Medhi, amedhi@macbook
* Last Modified time: 2017-05-21 17:40:34
* Copyright (C) Amal Medhi, amedhi@iisertvm.ac.in
*----------------------------------------------------------------------------*/
#include <iostream>
#include <iomanip>
#include <string>
#include <stdexcept>
#include <Eigen/SVD>
#include <Eigen/QR>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include "./stochastic_reconf.h"

namespace vmc {

int StochasticReconf::init(const input::Parameters& inputs, const VMC& vmc) 
{
  // problem size
  num_parms_ = vmc.num_varp();
  vparms_.resize(num_parms_);
  vparms_start_.resize(num_parms_);
  lbound_ = vmc.varp_lbound();
  ubound_ = vmc.varp_ubound();
  range_ = ubound_-lbound_;
  grad_.resize(num_parms_);
  sr_matrix_.resize(num_parms_,num_parms_);

  // optimization parameters
  int nowarn;
  num_sim_samples_ = inputs.set_value("sr_measure_steps", -1, nowarn);
  num_opt_samples_ = inputs.set_value("sr_opt_samples", 10, nowarn);
  max_iter_ = inputs.set_value("sr_max_iter", 200, nowarn);
  start_tstep_ = inputs.set_value("sr_start_tstep", 0.05, nowarn);
  random_start_ = inputs.set_value("sr_random_start", true, nowarn);
  flat_tail_len_ = inputs.set_value("sr_flat_tail_len", 20, nowarn);
  rslope_tol_ = inputs.set_value("sr_rslope_tol", 1.0E-2, nowarn);
  aslope_tol_ = inputs.set_value("sr_aslope_tol", 1.0E-5, nowarn);
  //mk_series_len_ = inputs.set_value("sr_series_len", 40, nowarn);
  //stabilizer_ = inputs.set_value("sr_stabilizer", 1.0E-4, nowarn);
  //mk_thresold_ = inputs.set_value("sr_fluctuation_tol", 0.30, nowarn);
  grad_tol_ = inputs.set_value("sr_grad_tol", 0.01, nowarn);
  print_progress_ = inputs.set_value("sr_progress_stdout", false, nowarn);
  print_log_ = inputs.set_value("sr_progress_log", true, nowarn);

  // optimal parameter values
  std::string mode = inputs.set_value("mode", "NEW");
  boost::to_upper(mode);
  bool replace_mode = true;
  if (mode=="APPEND") replace_mode = false;
  optimal_parms_.init("opt_params");
  optimal_parms_.set_ofstream(vmc.prefix_dir());
  optimal_parms_.resize(vmc.num_varp(), vmc.varp_names());
  optimal_parms_.set_replace_mode(replace_mode);
  optimal_parms_.switch_on();
  optimal_parms_.switch_on();
  std::vector<std::string> elem_names{"energy"};
  optimal_energy_.init("opt_energy");
  optimal_energy_.set_ofstream(vmc.prefix_dir());
  optimal_energy_.resize(1, elem_names);
  optimal_energy_.set_replace_mode(replace_mode);
  optimal_energy_.switch_on();
  energy_error_bar_.init("error_bar");
  energy_error_bar_.switch_on();

  // observable file header
  std::stringstream heading;
  vmc.print_info(heading);
  optimal_parms_.print_heading(heading.rdbuf()->str(), vmc.xvar_names());
  optimal_energy_.print_heading(heading.rdbuf()->str(), vmc.xvar_names());
  xvar_values_ = vmc.xvar_values();

  // iteration related file names
  life_fname_ = vmc.prefix_dir()+"ALIVE.d";
  iter_efile_ = vmc.prefix_dir()+"iter_energy.txt";
  iter_vfile_ = vmc.prefix_dir()+"iter_params.txt";

  // progress file
  if (print_log_) {
    logfile_.open(vmc.prefix_dir()+"log_optimization.txt");
    if (!logfile_.is_open())
      throw std::runtime_error("StochasticReconf::init: file open failed");
    vmc.print_info(logfile_);
    logfile_ << "#" << std::string(72, '-') << std::endl;
    logfile_ << "Stochastic Reconfiguration" << std::endl;
    logfile_ << "num_opt_samples = " << num_opt_samples_ << std::endl;
    logfile_ << "max_iter = " << max_iter_ << std::endl;
    logfile_ << "random_start = " << random_start_ << std::endl;
    logfile_ << "start_tstep = " << start_tstep_ << std::endl;
    logfile_ << "stabilizer = " << stabilizer_ << std::endl;
    logfile_ << "flat_tail_len = " << flat_tail_len_ << std::endl;
    logfile_ << "rslope_tol = " << rslope_tol_ << std::endl;
    logfile_ << "aslope_tol = " << aslope_tol_ << std::endl;
    //logfile_ << "grad_tol = " << grad_tol_ << std::endl;
    //logfile_ << "optimization samples = " << num_opt_samples_ << std::endl;
    logfile_ << "#" << std::string(72, '-') << std::endl;
  }
  if (print_progress_) {
    std::cout << "#" << std::string(72, '-') << std::endl;
    std::cout << "Stochastic Reconfiguration" << std::endl;
    std::cout << "num_opt_samples = " << num_opt_samples_ << std::endl;
    std::cout << "max_iter = " << max_iter_ << std::endl;
    std::cout << "start_tstep = " << start_tstep_ << std::endl;
    std::cout << "random_start = " << random_start_ << std::endl;
    std::cout << "stabilizer = " << stabilizer_ << std::endl;
    std::cout << "flat_tail_len = " << flat_tail_len_<< std::endl;
    std::cout << "rslope_tol = " << rslope_tol_ << std::endl;
    std::cout << "aslope_tol = " << aslope_tol_ << std::endl;
    //std::cout << "grad_tol = " << grad_tol_ << std::endl;
    //std::cout << "optimization samples = " << num_opt_samples_ << std::endl;
    std::cout << "#" << std::string(72, '-') << std::endl;
  }
  return 0;
}

int StochasticReconf::optimize(VMC& vmc)
{
  /*
  //std::vector<double> en = {0.3736,2.4375,3.9384,3.3123,5.4947,5.4332,6.3932,9.0605,9.3642,9.5207};
  std::vector<double> en = {-0.0546347,-0.0555617,-0.0559485,-0.0555145,-0.0560293,-0.0566849,
    -0.0571339,-0.0576311,-0.0584373,-0.0582745,-0.0586636,-0.0585779,
    -0.0587886,-0.0589912,-0.0589437,-0.0596863,-0.0593831,-0.0597698,
    -0.0593647,-0.0593304,-0.0603898,-0.0599514,-0.0592811,-0.0596215};
  if_converged(en);
  */
  xvar_values_ = vmc.xvar_values();
  optimal_parms_.reset();
  optimal_energy_.reset();
  energy_error_bar_.reset();
  // start optimization
  // starting value of variational parameters
  int refinement_cycle = 50;
  vparms_start_ = vmc.varp_values();
  bool all_samples_converged = true;
  for (int sample=1; sample<=num_opt_samples_; ++sample) {
    // iteration files
    std::ofstream life_fs(life_fname_);
    life_fs.close();
    file_energy_.open(iter_efile_);
    file_vparms_.open(iter_vfile_);
    vmc.print_info(file_energy_);
    file_energy_<<std::scientific<<std::uppercase<<std::setprecision(6)<<std::right;
    file_vparms_<<std::scientific<<std::uppercase<<std::setprecision(6)<<std::right;
    file_energy_ << "# Results: " << "Iteration Energy";
    file_energy_ << " (sample "<<sample<<" of "<<num_opt_samples_<<")\n";
    file_energy_ << "#" << std::string(72, '-') << "\n";
    file_energy_ << "# ";
    file_energy_ << std::left;
    file_energy_ <<std::setw(7)<<"iter"<<std::setw(15)<<"energy"<<std::setw(15)<<"err";
    file_energy_ << std::endl;
    file_energy_ << "#" << std::string(72, '-') << "\n";
    file_energy_ << std::right;

    vmc.print_info(file_vparms_);
    file_vparms_ << "# Results: " << "Iteration Variational Parameters";
    file_vparms_ << " (sample "<<sample<<" of "<<num_opt_samples_<<")\n";
    file_vparms_ << "#" << std::string(72, '-') << "\n";
    file_vparms_ << "# ";
    file_vparms_ << std::left;
    file_vparms_ << std::setw(7)<<"iter";
    for (const auto& p : vmc.varp_names()) file_vparms_<<std::setw(15)<<p.substr(0,14);
    file_vparms_ << std::endl;
    file_vparms_ << "#" << std::string(72, '-') << "\n";
    file_vparms_ << std::right;

    // message
    if (print_progress_) {
      std::cout << " Starting sample "<<sample<<" of "<<num_opt_samples_<<"\n"; 
      std::cout << std::flush;
    }
    if (print_log_) {
      logfile_ << " Starting sample "<<sample<<" of "<<num_opt_samples_<<"\n"; 
      logfile_ << std::flush;
    }

    // give gaussian randomness to the parameters
    if (random_start_) {
      for (int i=0; i<num_parms_; ++i) {
        double sigma = 0.10*(ubound_[i]-lbound_[i]);
        std::normal_distribution<double> random_normal(vparms_start_[i],sigma);
        double x = random_normal(vmc.rng());
        if (x>=lbound_[i] && x<=ubound_[i]) {
          vparms_[i] = x;
        }
        else {
          vparms_[i] = vparms_start_[i];
        }
      }
    }
    else {
      vparms_ = vparms_start_;
    }
    // seed vmc rng
    int rng_seed = -1;
    //vmc.seed_rng(1);
    // initial function values
    double start_en, error_bar; 
    vmc.sr_function(vparms_,start_en,error_bar,grad_,sr_matrix_,num_sim_samples_,rng_seed);
    // Stochastic reconfiguration iterations
    bool converged = false;
    std::vector<double> iter_energy;
    std::vector<double> iter_energy_err;
    std::deque<mcdata::data_t> iter_vparms;
    std::deque<mcdata::data_t> iter_grads;
    RealVector vparm_slope(num_parms_);
    RealVector grads_slope(num_parms_);
    double tstep = start_tstep_;
    int rcount = refinement_cycle;
    for (int iter=1; iter<=max_iter_; ++iter) {
      // apply to stabilizer to sr matrix 
      for (int i=0; i<num_parms_; ++i) sr_matrix_(i,i) += stabilizer_;
      // search direction
      Eigen::VectorXd search_dir = sr_matrix_.fullPivLu().solve(-grad_);
      //Eigen::VectorXd search_dir = sr_matrix_.inverse()*(-grad_);
      // update variables
      vparms_.noalias() += tstep * search_dir;
      // box constraint and max_norm (of components not hitting boundary) 
      //vparms_ = lbound_.cwiseMax(vparms_.cwiseMin(ubound_));
      /*
      for (int n=0; n<num_parms_; ++n) {
        double x = vparms_[n];
        if (x < lbound_[n]) vparms_[n] = lbound_[n];
        if (x > ubound_[n]) vparms_[n] = ubound_[n];
      }
      */

      // simulation at new parameters
      double en;
      vmc.sr_function(vparms_,en,error_bar,grad_,sr_matrix_,num_sim_samples_,rng_seed);
      double gnorm = grad_.squaredNorm();
      // file outs
      file_energy_<<std::setw(6)<<iter<<std::scientific<<std::setw(16)<<en; 
      file_energy_<<std::fixed<<std::setw(10)<<error_bar<<std::endl<<std::flush;
      file_vparms_<<std::setw(6)<<iter; 
      for (int i=0; i<vparms_.size(); ++i) file_vparms_<<std::setw(15)<<vparms_[i];
      file_vparms_<<std::endl<<std::flush;
      if (print_progress_) {
        std::ios state(NULL);
        state.copyfmt(std::cout);
        std::cout << "#" << std::string(60, '-') << std::endl;
        std::cout << std::left;
        std::cout << " iteration  =   "<<std::setw(6)<<iter<<"\n";
        std::cout <<std::scientific<<std::uppercase<<std::setprecision(6)<<std::right;
        /*
        std::cout << " grad       =";
        for (int i=0; i<grad_.size(); ++i)
          std::cout << std::setw(15)<<grad_[i];
        std::cout << "\n";
        std::cout << " search_dir =";
        for (int i=0; i<search_dir.size(); ++i)
          std::cout << std::setw(15)<<search_dir[i];
        std::cout << "\n";
        std::cout << " varp       =";
        for (int i=0; i<vparms_.size(); ++i)
          std::cout << std::setw(15)<<vparms_[i];
        std::cout << "\n";
        */
        std::cout << " gnorm      ="<<std::setw(15)<< gnorm<<"\n";
        std::cout << " energy     ="<<std::setw(15)<<en << "   (+/-) ";
        std::cout <<std::fixed<<std::setw(10)<<error_bar<<"\n";
        std::cout.copyfmt(state);
      }
      if (print_log_) {
        logfile_ << "#" << std::string(60, '-') << std::endl;
        logfile_ << std::left;
        logfile_ << " iteration  =   "<<std::setw(6)<<iter<<"\n";
        logfile_ <<std::scientific<<std::uppercase<<std::setprecision(6)<<std::right;
        logfile_ << " grad       =";
        for (int i=0; i<grad_.size(); ++i)
          logfile_ << std::setw(15)<<grad_[i];
        logfile_ << "\n";
        logfile_ << " search_dir =";
        for (int i=0; i<search_dir.size(); ++i)
          logfile_ << std::setw(15)<<search_dir[i];
        logfile_ << "\n";
        logfile_ << " varp       =";
        for (int i=0; i<vparms_.size(); ++i)
          logfile_ << std::setw(15)<<vparms_[i];
        logfile_ << "\n";
        logfile_ << " gnorm      ="<<std::setw(15)<< gnorm<<"\n";
        logfile_ << " energy     ="<<std::setw(15)<<en << "   (+/-) "; 
        logfile_ << std::fixed<<std::setw(10)<<error_bar<<"\n";
      }
      // check convergence
      iter_energy.push_back(en);
      iter_energy_err.push_back(error_bar);
      iter_vparms.push_back(vparms_);
      iter_grads.push_back(grad_);
      //if (false) {
      if (iter_energy.size() > flat_tail_len_) {
        iter_vparms.pop_front();
        iter_grads.pop_front();
        RealVector p = lsqfit(iter_energy,flat_tail_len_);
        fit_vparms(iter_vparms, vparm_slope);
        iter_mean(iter_grads, grads_slope);
        //std::cout << vparm_slope.transpose() << "\n"; // getchar();
        //std::cout << grads_slope.transpose() << "\n"; // getchar();
        double slope = std::abs(p(1));
        converged = false;
        if (slope<rslope_tol_*std::abs(start_en-en) || slope<aslope_tol_) {
	        rcount++;
          for (int n=0; n<num_parms_; ++n) {
            double vslope = std::abs(vparm_slope(n));
            double gslope = std::abs(grads_slope(n));
            if (vslope<rslope_tol_*range_(n)||vslope<aslope_tol_||gslope<0.1*rslope_tol_) {
              converged = true;
            }
            else {
              converged = false;
              break;
            }
          }
	  // reduce stepsize
	  if (rcount >= refinement_cycle) {
	    tstep *= 0.5;
	    rcount = 0;
	  }
        }
        if (converged) break;
      }
      // check life
      if (!boost::filesystem::exists(life_fname_)) break;
    }
    if (!converged) all_samples_converged = false;

    if (print_progress_) {
      std::cout << "#" << std::string(60, '-') << std::endl;
      if (converged) std::cout<<" Converged!"<<std::endl;
      else std::cout <<" NOT converged"<<std::endl;
      std::cout << "#" << std::string(60, '-') << std::endl << std::flush;
    }
    if (print_log_) {
      logfile_ << "#" << std::string(60, '-') << std::endl;
      if (converged) logfile_<<" Converged!"<<std::endl;
      else logfile_ <<" NOT converged"<<std::endl;
      logfile_ << "#" << std::string(60, '-')<<std::endl<<std::endl<< std::flush;
    }
    file_energy_.close();
    file_vparms_.close();

    // optimized quantities for this sample
    if (iter_vparms.size()>=flat_tail_len_) {
      int n = iter_energy.size()-flat_tail_len_;
      for (int i=0; i<flat_tail_len_; ++i) {
        optimal_energy_ << iter_energy[n+i];
        energy_error_bar_ << iter_energy_err[n+i];
        optimal_parms_ << iter_vparms[i];
      }
    }

    // print sample values
    // optimal energy
    optimal_energy_.open_file();
    optimal_energy_.fs()<<std::right;
    optimal_energy_.fs()<<std::scientific<<std::uppercase<<std::setprecision(6);
    optimal_energy_.fs()<<"#";
    for (const auto& p : xvar_values_) 
      optimal_energy_.fs()<<std::setw(14)<<p;
    optimal_energy_.fs()<<std::setw(15)<<optimal_energy_.mean();
    optimal_energy_.fs()<<std::fixed<<std::setw(10)<<energy_error_bar_.mean();
    optimal_energy_.fs()<<std::setw(10)<<num_sim_samples_; 
    if (converged) 
      optimal_energy_.fs()<<std::setw(11)<<"CONVERGED"<<std::setw(7)<<0<<std::endl;
    else 
      optimal_energy_.fs()<<std::setw(11)<<"NOT_CONVD"<<std::setw(7)<<0<<std::endl;
    optimal_energy_.close_file();
    //optimal_energy_.print_result(xvar_values_);
    // optimal variational parameters
    optimal_parms_.open_file();
    optimal_parms_.fs()<<"#";
    optimal_parms_.print_result(xvar_values_);

    // save the values
    optimal_energy_.save_result();
    energy_error_bar_.save_result();
    optimal_parms_.save_result();

    // reset for next sample
    optimal_energy_.reset();
    energy_error_bar_.reset();
    optimal_parms_.reset();
  } // samples
  logfile_.close();
  // print results

  // grand averages
  // optimal energy
  optimal_energy_.open_file();
  optimal_energy_.fs()<<std::right;
  optimal_energy_.fs()<<std::scientific<<std::uppercase<<std::setprecision(6);
  optimal_energy_.fs() << "#" << std::string(72, '-') << "\n";
  optimal_energy_.fs() << "# grand average:\n"; // 
  optimal_energy_.fs() << "#" << std::string(72, '-') << "\n";
  for (const auto& p : xvar_values_) 
    optimal_energy_.fs()<<std::setw(14)<<p;
  optimal_energy_.fs()<<std::setw(15)<<optimal_energy_.grand_data().mean();
  optimal_energy_.fs()<<std::fixed<<std::setw(10)<<energy_error_bar_.grand_data().mean();
  optimal_energy_.fs()<<std::setw(10)<<num_sim_samples_; 
  if (all_samples_converged) 
    optimal_energy_.fs()<<std::setw(11)<<"CONVERGED"<<std::setw(7)<<0<<std::endl;
  else 
    optimal_energy_.fs()<<std::setw(11)<<"NOT_CONVD"<<std::setw(7)<<0<<std::endl;
  optimal_energy_.close_file();
  optimal_parms_.print_grand_result(xvar_values_);
  // reset
  optimal_energy_.reset_grand_data();
  energy_error_bar_.reset_grand_data();
  optimal_parms_.reset_grand_data();

  return 0;
}

RealVector StochasticReconf::lsqfit(const std::vector<double>& iter_energy, 
  const int& max_fitpoints) const
{
  /* 
     Least Square Fit with 'f(x) = p0 + p1*x' (by QR method) 
  */
  int num_points = max_fitpoints;
  if (num_points > iter_energy.size()) num_points = iter_energy.size();

  //  The 'xrange' is set at [0:1.0]
  double dx = 1.0/num_points;
  RealMatrix A(num_points,2);
  for (int i=0; i<num_points; ++i) {
    A(i,0) = 1.0;
    A(i,1) = (i+1) * dx;
  }
  // take the 'last num_points' values
  int n = iter_energy.size()-num_points;
  RealVector b(num_points);
  for (int i=0; i<num_points; ++i) b[i] = iter_energy[n+i];
  RealVector p = A.colPivHouseholderQr().solve(b);
  //std::cout << "poly = " << p.transpose() << "\n"; 
  //std::cout << "slope = " << p(1) << "\n"; 
  //getchar();
  return p;
}

void StochasticReconf::fit_vparms(const std::deque<mcdata::data_t>& iter_vparms, RealVector& fit_slope) const
{
  int num_points = iter_vparms.size();
  //  The 'xrange' is set at [0:1.0]
  double dx = 1.0/num_points;
  RealMatrix A(num_points,2);
  for (int i=0; i<num_points; ++i) {
    A(i,0) = 1.0;
    A(i,1) = (i+1) * dx;
  }
  for (int n=0; n<num_parms_; ++n) {
    RealVector b(num_points);
    for (int i=0; i<num_points; ++i) b[i] = iter_vparms[i][n];
    RealVector p = A.colPivHouseholderQr().solve(b);
    fit_slope(n) = p(1);
  }
} 

void StochasticReconf::iter_mean(const std::deque<mcdata::data_t>& iter_vparms, RealVector& mean) const
{
  int num_points = iter_vparms.size();
  mean.setZero();
  for (auto& elem : iter_vparms) {
    for (int i=0; i<num_parms_; ++i) {
      mean(i) += elem(i);
    }
  }
  for (int i=0; i<num_parms_; ++i) {
    mean(i) /= num_points;
  }
} 

RealVector StochasticReconf::lsqfit_poly(const std::vector<double>& iter_energy, 
  const int& poly_deg) const
{
  /* 
     Least Square Fit (by QR method) of the data with 
     polynomial of degree 'fitpoly_degree_'
  */
  int num_points = iter_energy.size();
  int poly_degree = 3;
  //  The 'xrange' is set at [0:1.0]
  double dx = 1.0/num_points;
  RealMatrix A(num_points,2);
  for (int i=0; i<num_points; ++i) {
    A(i,0) = 1.0;
    double x = (i+1) * dx;
    for (int n=1; n<=poly_degree; ++n) A(i,n) = A(i,n-1)*x;
  }
  RealVector b(num_points);
  for (int i=0; i<num_points; ++i) b[i] = iter_energy[i];
  RealVector p = A.colPivHouseholderQr().solve(b);
  return p;
}




} // end namespace vmc
