/*---------------------------------------------------------------------------
* Author: Amal Medhi
* Date:   2017-03-09 15:19:43
* Last Modified by:   Amal Medhi, amedhi@macbook
* Last Modified time: 2017-05-21 17:40:34
* Copyright (C) Amal Medhi, amedhi@iisertvm.ac.in
*----------------------------------------------------------------------------*/
#include <iostream>
#include <limits>
#include <iomanip>
#include <string>
#include <stdexcept>
#include <Eigen/SVD>
#include <Eigen/QR>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include "./optimizer.h"

namespace vmc {

int Optimizer::init(const input::Parameters& inputs, const VMCRun& vmc) 
{
  // Number of variational parameters & bounds
  num_parms_ = vmc.num_varparms();
  num_parms_print_ = std::min(num_parms_,10);
  varp_names_.resize(num_parms_);
  varp_lbound_.resize(num_parms_);
  varp_ubound_.resize(num_parms_);
  varp_bounded_.resize(num_parms_);
  vmc.get_varp_names(varp_names_);
  vmc.get_varp_lbound(varp_lbound_);
  vmc.get_varp_ubound(varp_ubound_);
  varp_bound_exists_ = false;
  for (int n=0; n<num_parms_; ++n) {
    if (std::isinf(-varp_lbound_[n]) && std::isinf(varp_ubound_[n])) {
      varp_bounded_[n] = 0;
    }
    else {
      varp_bounded_[n] = 1;
      varp_bound_exists_ = true;
    }
  }

  // optimization parameters
  int nowarn;
  num_opt_samples_ = inputs.set_value("opt_num_samples", 10, nowarn);
  maxiter_ = inputs.set_value("opt_maxiter", 100, nowarn);
  max_ls_steps_ = inputs.set_value("opt_ls_steps", 0, nowarn);
  CG_maxiter_ = inputs.set_value("opt_cg_maxiter", 0, nowarn);
  CG_alpha0_ = inputs.set_value("opt_cg_alpha0", 0.1, nowarn);
  if (CG_maxiter_>0) {
    std::string str = inputs.set_value("opt_cg_algorithm", "PR", nowarn);
    boost::to_upper(str);
    if (str=="FR") {
      CG_Algorithm_ = CG_type::FR;
    }
    else if (str=="PR") {
      CG_Algorithm_ = CG_type::PR;
    }
    else if (str=="DY") {
      CG_Algorithm_ = CG_type::DY;
    }
    else if (str=="HS") {
      CG_Algorithm_ = CG_type::HS;
    }
    else {
      throw std::range_error("Optimizer::init: invalid CG method");
    }
  }
  // Stochastic Reconf
  std::string solver = inputs.set_value("opt_sr_solver", "BDCSVD", nowarn);
  boost::to_upper(solver);
  if (solver=="BDCSVD") {
    SR_solver_ = SR_solver::BDCSVD;
  }
  else if (solver=="JACOBISVD") {
    SR_solver_ = SR_solver::JacobiSVD;
  }
  else if (solver=="EIGSOLVER") {
    SR_solver_ = SR_solver::EigSolver;
  }
  else {
    throw std::range_error("Optimizer::init: unknown SR_solver");
  }
  start_tstep_ = inputs.set_value("opt_start_tstep", 0.1, nowarn);
  refinement_cycle_ = inputs.set_value("opt_refinement_cycle", 40, nowarn);
  stabilizer_ = inputs.set_value("opt_stabilizer", 0.2, nowarn);
  sr_diag_shift_ = inputs.set_value("opt_diag_shift", 0.001, nowarn);
  sr_diag_scale_ = inputs.set_value("opt_diag_scale", 0.01, nowarn);
  w_svd_cut_ = inputs.set_value("opt_svd_cut", 0.001, nowarn);
  //random_start_ = inputs.set_value("sr_random_start", false, nowarn);
  grad_tol_ = inputs.set_value("opt_gradtol", 5.0E-3, nowarn);
  print_progress_ = inputs.set_value("opt_progress_stdout", false, nowarn);
  print_log_ = inputs.set_value("opt_progress_log", true, nowarn);

  // probLS parameters
  num_probls_steps_ = inputs.set_value("opt_pls_steps", 0, nowarn);
  pls_c1_ = inputs.set_value("opt_pls_c1", 0.05, nowarn);
  pls_cW_ = inputs.set_value("opt_pls_cW", 0.3, nowarn);
  pls_alpha0_ = inputs.set_value("opt_pls_alpha0", 0.02, nowarn);
  //pls_target_df_ = inputs.set_value("opt_pls_target_df", 0.5, nowarn);
  probLS_.set_c1(pls_c1_);
  probLS_.set_cW(pls_cW_);
  probLS_.set_alpha0(pls_alpha0_);
  //probLS_.set_target_df(pls_target_df_);

  // Mann-Kendall statistic
  mk_series_len_ = inputs.set_value("opt_mkseries_len", 20, nowarn);
  mk_thresold_ = inputs.set_value("opt_mkfluct_tol", 0.30, nowarn);
  mk_statistic_.resize(num_parms_);
  mk_statistic_en_.resize(1); // for energy
  mk_statistic_.set_maxlen(mk_series_len_);
  mk_statistic_en_.set_maxlen(std::min(6,mk_series_len_));

  // Observables for optimal parameters
  num_vmc_samples_ = vmc.num_measure_steps();
  std::string mode = inputs.set_value("mode", "NEW");
  boost::to_upper(mode);
  bool replace_mode = true;
  if (mode=="APPEND") replace_mode = false;
  optimal_parms_.init("opt_params");
  optimal_parms_.set_ofstream(vmc.prefix_dir());
  optimal_parms_.resize(num_parms_,varp_names_);
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

  // iteration related files 
  boost::filesystem::path iter_prefix(vmc.prefix_dir()+"/iterations");
  boost::filesystem::create_directory(iter_prefix);
  life_fname_ = vmc.prefix_dir()+"/iterations/ALIVE.d";
  iter_efile_ = vmc.prefix_dir()+"/iterations/iter_energy";
  iter_vfile_ = vmc.prefix_dir()+"/iterations/iter_params";

  return 0;
}

int Optimizer::print_info(const VMCRun& vmc)
{
  if (print_log_) {
    logfile_.open(vmc.prefix_dir()+"log_optimization.txt");
    if (!logfile_.is_open())
      throw std::runtime_error("Optimizer::init: file open failed");
    vmc.print_info(logfile_);
    logfile_ << "#" << std::string(72, '-') << std::endl;
    logfile_ << "Stochastic Reconfiguration" << std::endl;
    logfile_ << "num_opt_samples = " << num_opt_samples_ << std::endl;
    logfile_ << "max_iter = " << maxiter_ << std::endl;
    //logfile_ << "random_start = " << random_start_ << std::endl;
    logfile_ << "stabilizer = " << stabilizer_ << std::endl;
    logfile_ << "start_tstep = " << start_tstep_ << std::endl;
    logfile_ << "w_svd_cut = " << w_svd_cut_ << std::endl;
    logfile_ << "grad_tol = " << grad_tol_ << std::endl;
    //logfile_ << "ftol = " << ftol_ << std::endl;
    logfile_ << "#" << std::string(72, '-') << std::endl;
  }
  if (print_progress_) {
    std::cout << "#" << std::string(72, '-') << std::endl;
    std::cout << "Stochastic Reconfiguration" << std::endl;
    std::cout << "num_opt_samples = " << num_opt_samples_ << std::endl;
    std::cout << "max_iter = " << maxiter_ << std::endl;
    //std::cout << "random_start = " << random_start_ << std::endl;
    std::cout << "stabilizer = " << stabilizer_ << std::endl;
    std::cout << "start_tstep = " << start_tstep_ << std::endl;
    std::cout << "w_svd_cut = " << w_svd_cut_ << std::endl;
    std::cout << "grad_tol = " << grad_tol_ << std::endl;
    //std::cout << "ftol = " << ftol_ << std::endl;
    std::cout << "#" << std::string(72, '-') << std::endl;
  }
  return 0;
}

int Optimizer::init_sample(const VMCRun& vmc, const int& sample)
{
  // reset parameters
  search_tstep_ = start_tstep_;
  fixedstep_iter_ = 1;
  refinement_level_ = 0;

  // Mann-Kendall statistic for converegence test
  mk_statistic_.reset();
  mk_statistic_en_.reset();
  iter_parms_.clear();
  iter_energy_.clear();
  iter_energy_err_.clear();
  iter_gnorm_.clear();

  // iteration files
  std::ostringstream suff;
  suff << "_sample"<<std::setfill('0')<<std::setw(2)<<sample<<".txt";
  file_life_.open(life_fname_);
  file_life_.close();
  file_energy_.open(iter_efile_+suff.str());
  file_vparms_.open(iter_vfile_+suff.str());

  // init Observables for optimal energy & parameters
  xvar_values_ = vmc.xvar_values();
  optimal_parms_.reset();
  optimal_energy_.reset();
  energy_error_bar_.reset();

  vmc.print_info(file_energy_);
  file_energy_<<std::scientific<<std::uppercase<<std::setprecision(6)<<std::right;
  file_vparms_<<std::scientific<<std::uppercase<<std::setprecision(6)<<std::right;
  file_energy_ << "# Results: " << "Iteration Energy";
  file_energy_ << " (sample "<<sample<<" of "<<num_opt_samples_<<")\n";
  file_energy_ << "#" << std::string(72, '-') << "\n";
  file_energy_ << "# ";
  file_energy_ << std::left;
  file_energy_ <<std::setw(7)<<"iter"<<std::setw(15)<<"energy"<<std::setw(15)<<"err";
  file_energy_ << "\n";
  file_energy_ << "#" << std::string(72, '-') << "\n";
  file_energy_ << std::right;

  vmc.print_info(file_vparms_);
  file_vparms_ << "# Results: " << "Iteration Variational Parameters";
  file_vparms_ << " (sample "<<sample<<" of "<<num_opt_samples_<<")\n";
  file_vparms_ << "#" << std::string(72, '-') << "\n";
  file_vparms_ << "# ";
  file_vparms_ << std::left;
  file_vparms_ << std::setw(7)<<"iter";
  for (const auto& p : varp_names_) file_vparms_<<std::setw(15)<<p.substr(0,14);
  file_vparms_ << "\n";
  file_vparms_ << "#" << std::string(72, '-') << "\n";
  file_vparms_ << std::right;
  // message
  if (print_progress_) {
    std::cout << " Starting sample "<<sample<<" of "<<num_opt_samples_<<"\n"; 
    //std::cout << "#" << std::string(60, '-') << std::endl;
    std::cout << std::flush;
  }
  if (print_log_) {
    logfile_ << " Starting sample "<<sample<<" of "<<num_opt_samples_<<"\n"; 
    //logfile_ << "#" << std::string(60, '-') << std::endl;
    logfile_ << std::flush;
  }
  return 0;
}

int Optimizer::optimize(VMCRun& vmc)
{
  // Observables
  xvar_values_ = vmc.xvar_values();
  optimal_parms_.reset();
  optimal_energy_.reset();
  energy_error_bar_.reset();

  // start optimization
  double en_trend;
  double sq_gnorm, sq_gnorm_prev, gnorm, gnorm1, avg_gnorm;
  RealVector vparms(num_parms_);
  RealVector vparms_start(num_parms_);
  RealVector grad(num_parms_);
  RealVector grad_err(num_parms_);
  RealVector search_dir(num_parms_);
  RealMatrix sr_matrix(num_parms_,num_parms_);
  RealMatrix work_mat(num_parms_,num_parms_);
  double en, en_err;
  bool silent = true;


  if (num_opt_samples_>1) {
    vmc.get_varp_values(vparms_start);
  }
  else {
    vmc.get_varp_values(vparms);
    //std::cout << vparms.transpose() << "\n"; getchar();
  }

  // for all samples
  all_converged_ = true;
  for (int sample=1; sample<=num_opt_samples_; ++sample) {

    // initialize quantities for this sample
    init_sample(vmc, sample);

    // reset MK statistics
    mk_statistic_.reset();
    mk_statistic_en_.reset();

    // starting parameters (for 1 sample, already initialized)
    if (num_opt_samples_>1) vparms = vparms_start;

    // iteration count
    int iter_count = 0;
    exit_status status = exit_status::notconvgd;

   /*--------------------------------------------------------------
    * Stochastic CG iteraions (NEW)
    *--------------------------------------------------------------*/
    if (CG_maxiter_>0) {

      // 0-th iteration
      double CG_beta = 0.0;
      RealVector previous_grad(num_parms_);

      // compute initial quantities 
      vmc.run(vparms,en,en_err,grad,grad_err,silent);
      sq_gnorm = grad.squaredNorm();
      gnorm = std::sqrt(sq_gnorm);
      gnorm1 = gnorm/num_parms_;
      avg_gnorm = gnorm1;
      search_dir = -grad;
      en_trend = 1;

      // CG iterations
      bool conv_condition_reached = false;
      int final_iter_count = 0;
      for (int cg_iter=1; cg_iter<=CG_maxiter_; ++cg_iter) {
        iter_count++;

        // message
        if (print_progress_) {
          print_progress(std::cout, iter_count, "Stochastic CG");
          print_progress(std::cout,vparms,en,en_err,grad,search_dir,
            gnorm1,avg_gnorm,en_trend);
          std::cout<<" beta        =  " << CG_beta << "\n";
          //std::cout<<" line search =  adaptive step\n";
        }
        if (print_log_) {
          print_progress(logfile_, iter_count, "Stochastic CG");
          print_progress(logfile_,vparms,en,en_err,grad,search_dir,
            gnorm1,avg_gnorm,en_trend);
          logfile_ <<" beta        =  " << CG_beta << "\n";
          //logfile_<<" line search =  constant step\n";
        }

        // MK test: add data to Mann-Kendall statistic
        mk_statistic_en_ << en;
        iter_energy_.push_back(en);
        iter_energy_err_.push_back(en_err);
        iter_gnorm_.push_back(gnorm1);
        if (mk_statistic_en_.is_full()) {
          iter_energy_.pop_front();
          iter_energy_err_.pop_front();
          iter_gnorm_.pop_front();
        }
        en_trend = mk_statistic_en_.max_trend();
        // series average of gnorm
        avg_gnorm = series_avg(iter_gnorm_);

        // file outs
        file_energy_<<std::setw(6)<<iter_count<<std::scientific<<std::setw(16)<<en; 
        file_energy_<<std::fixed<<std::setw(10)<<en_err<<std::endl<<std::flush;
        file_vparms_<<std::setw(6)<<iter_count; 
        for (int i=0; i<num_parms_print_; ++i) file_vparms_<<std::setw(15)<<vparms[i];
        file_vparms_<<std::endl<<std::flush;

        /*--------------------------------------------------------------
         * Convergence criteria
        *--------------------------------------------------------------*/
        if (mk_statistic_en_.is_full()) {
          if (en_trend<=mk_thresold_ && avg_gnorm<=grad_tol_) {
            conv_condition_reached = true;
          }
        }
        if (conv_condition_reached) {
          iter_parms_.push_back(vparms);
          if (print_progress_) {
            std::cout<<" final iter  =  "<<final_iter_count<<"/"<<mk_series_len_<<"\n";
          } 
          if (print_log_) {
            logfile_<<" final iter  =  "<<final_iter_count<<"/"<<mk_series_len_<<"\n";
          }
          final_iter_count++;
          if (final_iter_count >= mk_series_len_) {
            //mk_statistic_.get_series_avg(vparms);
            status = exit_status::converged;
            break;
          }
        }
        if (cg_iter > CG_maxiter_) {
          iter_parms_.push_back(vparms);
          status = exit_status::maxiter;
          break;
        }
        if (!boost::filesystem::exists(life_fname_)) {
          iter_parms_.push_back(vparms);
          status = exit_status::terminated;
          break;
        }


        // copy previous grad
        previous_grad = grad;
        sq_gnorm_prev = sq_gnorm;

        // update parameters by backtracting Armjio
        bool success = backtracking_Armjio_step(vmc,vparms,en,en_err,grad,grad_err,search_dir); 
        if (!success) {
          search_dir = -grad;
          vparms.noalias() += search_tstep_ * search_dir;
          apply_projection(vparms);
          vmc.run(vparms,en,en_err,grad,grad_err,silent);
          if (print_progress_) {
            std::cout << " step size   =  "<< search_tstep_ <<"\n";
          }
          if (print_log_) {
            logfile_ << " step size   =  "<< search_tstep_ <<"\n";
          }
        }

        // update gradient norm
        sq_gnorm = squaredNormProj(grad, vparms);
        gnorm = std::sqrt(sq_gnorm);
        gnorm1 = gnorm/num_parms_;

        // update beta
        double g_dot_pg = grad.dot(previous_grad);
        if (CG_Algorithm_==CG_type::FR) {
          CG_beta = sq_gnorm/sq_gnorm_prev;
        }
        else if (CG_Algorithm_==CG_type::PR) {
          CG_beta = (sq_gnorm - grad.dot(previous_grad))/sq_gnorm_prev;
        }
        else if (CG_Algorithm_==CG_type::HS) {
          RealVector grad_diff = grad-previous_grad;
          CG_beta = grad.dot(grad_diff)/search_dir.dot(grad_diff);
        }
        else if (CG_Algorithm_==CG_type::DY) {
          RealVector grad_diff = grad-previous_grad;
          CG_beta = sq_gnorm/search_dir.dot(grad_diff);
        }
        else {
          throw std::range_error("Optimizer::stochastic_CG: unknown CG method");
        }
        // discard negative values
        CG_beta = std::max(CG_beta, 0.0);

        // update conjugate search direction
        search_dir = -grad + CG_beta*search_dir;

        // Powel restart condition
        if (g_dot_pg > 0.2*sq_gnorm) search_dir = -grad;
      }
    }


    // reset MK statistics
    mk_statistic_.reset();
    mk_statistic_en_.reset();

   /*--------------------------------------------------------------
    * Stochastic Reconfiguration iteraions 
    *--------------------------------------------------------------*/
    // counter for last data points around minimum for final average
    bool conv_condition_reached = false;
    int final_iter_count = 0;

    //exit_status status = exit_status::notconvgd;
    for (int sr_iter=1; sr_iter<=maxiter_; ++sr_iter) {
      iter_count++;

      // first message 
      if (print_progress_) print_progress(std::cout, iter_count, "SR");
      if (print_log_) print_progress(logfile_, iter_count, "SR");

     /*----------------------------------------------------------------
      * Search direction by Stochastic Reconfiguration
      *----------------------------------------------------------------*/
      vmc.run(vparms,en,en_err,grad,grad_err,sr_matrix,silent);
      stochastic_reconf(grad,sr_matrix,work_mat,search_dir);

      // max_norm (of components not hitting boundary) 
      //gnorm = std::sqrt(grad.squaredNorm());
      gnorm = std::sqrt(squaredNormProj(grad,vparms));
      gnorm1 = gnorm/num_parms_;
      //double proj_norm = projected_gnorm(vparms,grad,varp_lbound_,varp_ubound_);
      //gnorm = std::min(gnorm, proj_norm);

      // MK test: add data to Mann-Kendall statistic
      mk_statistic_en_ << en;
      iter_energy_.push_back(en);
      iter_energy_err_.push_back(en_err);
      iter_gnorm_.push_back(gnorm1);
      if (mk_statistic_en_.is_full()) {
        iter_energy_.pop_front();
        iter_energy_err_.pop_front();
        iter_gnorm_.pop_front();
      }
      // series average of gnorm
      avg_gnorm = series_avg(iter_gnorm_);
      // MK trend
      en_trend = mk_statistic_en_.max_trend();

      // file outs
      file_energy_<<std::setw(6)<<iter_count<<std::scientific<<std::setw(16)<<en; 
      file_energy_<<std::fixed<<std::setw(10)<<en_err<<std::endl<<std::flush;
      file_vparms_<<std::setw(6)<<iter_count; 
      for (int i=0; i<num_parms_print_; ++i) file_vparms_<<std::setw(15)<<vparms[i];
      file_vparms_<<std::endl<<std::flush;

      // print progress 
      if (print_progress_) {
        print_progress(std::cout,vparms,en,en_err,grad,search_dir,
          gnorm1,avg_gnorm,en_trend);
      }
      if (print_log_) {
        print_progress(logfile_,vparms,en,en_err,grad,search_dir,
          gnorm1,avg_gnorm,en_trend);
      }

     /*--------------------------------------------------------------
      * Convergence criteria
      *--------------------------------------------------------------*/
      if (mk_statistic_en_.is_full()) {
        if (en_trend<=mk_thresold_ && avg_gnorm<=grad_tol_) {
          conv_condition_reached = true;
        }
        //mk_statistic_.get_series_avg(vparms);
        //status = exit_status::converged;
        //break;
      }
      if (conv_condition_reached) {
        iter_parms_.push_back(vparms);
        if (print_progress_) {
          std::cout<<" final iter  =  "<<final_iter_count<<"/"<<mk_series_len_<<"\n";
        } 
        if (print_log_) {
          logfile_<<" final iter  =  "<<final_iter_count<<"/"<<mk_series_len_<<"\n";
        }
        final_iter_count++;
        if (final_iter_count >= mk_series_len_) {
          //mk_statistic_.get_series_avg(vparms);
          status = exit_status::converged;
          break;
        }
      }
      if (sr_iter>=maxiter_ || iter_count>= maxiter_) {
        iter_parms_.push_back(vparms);
        status = exit_status::maxiter;
        break;
      }
      if (!boost::filesystem::exists(life_fname_)) {
        iter_parms_.push_back(vparms);
        status = exit_status::terminated;
        break;
      }

      /*----------------------------------------------------------------
       * Step ahead by either Line Search OR by a fixed sized step. 
       *----------------------------------------------------------------*/
      bool success = false;
      if (max_ls_steps_>0) {
        // Line search
        //line_search(vmc,vparms,en,en_err,grad,grad_err,search_dir);
        success = backtracking_Armjio_step(vmc,vparms,en,en_err,grad,grad_err,search_dir); 
      }
      if (!success) {
        // update by fixed sized step 
        vparms.noalias() += search_tstep_ * search_dir;
        apply_projection(vparms);
        fixedstep_iter_++;
        if (print_progress_) {
          std::cout<<" line search =  constant step\n"; 
          std::cout<<" step size   =  "<<search_tstep_<<"\n"; 
        } 
        if (print_log_) {
          logfile_<<" line search =  constant step\n";
          logfile_<<" step size   =  "<<search_tstep_<<"\n";
        }
      }

      // refine step size
      if (fixedstep_iter_ % refinement_cycle_ == 0) {
        refinement_level_++;
        search_tstep_ *= 0.5;
      }

      // save snapshot of NQS parameter values
      if ((iter_count) % 50 == 0) {
        //std::cout << " >> Optimizer::optimize: Currently NOT saving parameters to file\n";
        vmc.save_parameters(vparms);
      }
    } // iterations

    // Iterations over for this sample, finalize
    if (print_progress_) {
      std::cout << "#" << std::string(60, '-') << std::endl;
      if (status == exit_status::converged) {
        std::cout<<" Converged!"<<std::endl;
      }
      else if (status == exit_status::maxiter) {
        std::cout<<" Iterations exceeded (NOT converged)"<<std::endl;
      }
      else if (status == exit_status::terminated) {
        std::cout<<" Iterations terminated (NOT converged)"<<std::endl;
      }
      else {
        std::cout <<" NOT converged"<<std::endl;
      }
      std::cout << "#" << std::string(60, '-') << std::endl << std::flush;
    }
    if (print_log_) {
      logfile_ << "#" << std::string(60, '-') << std::endl;
      if (status == exit_status::converged) {
        logfile_<<" Converged!"<<std::endl;
      }
      else if (status == exit_status::maxiter) {
        logfile_<<" Iterations exceeded (NOT converged)"<<std::endl;
      }
      else if (status == exit_status::terminated) {
        logfile_<<" Iterations terminated (NOT converged)"<<std::endl;
      }
      else {
        logfile_ <<" NOT converged"<<std::endl;
      }
      logfile_ << "#" << std::string(60, '-')<<std::endl<<std::endl<< std::flush;
    }
    if (status != exit_status::converged) {
      all_converged_ = false;
    }
    finalize_sample(status);
  }// end of samples

  // save final (averaged) NQS parameters
  vparms = optimal_parms_.grand_data().mean_data();
  //std::cout << " >> Optimizer::optimize: Currently NOT saving parameters to file\n";
  vmc.save_parameters(vparms);

  return 0;
}

int Optimizer::finalize_sample(const exit_status& status)
{
  file_energy_.close();
  file_vparms_.close();
  for (int i=0; i<iter_energy_.size(); ++i) {
    optimal_energy_ << iter_energy_[i];
    energy_error_bar_ << iter_energy_err_[i];
  }
  for (int i=0; i<iter_parms_.size(); ++i) {
    optimal_parms_ << iter_parms_[i];
  }
  // print sample values
  // optimal energy
  optimal_energy_.open_file();
  optimal_energy_.fs()<<std::right;
  optimal_energy_.fs()<<std::scientific<<std::uppercase<<std::setprecision(6);
  optimal_energy_.fs()<<"#";
  for (const auto& p : xvar_values_) 
    optimal_energy_.fs()<<std::setw(13)<<p;
  optimal_energy_.fs()<<std::setw(15)<<optimal_energy_.mean();
  optimal_energy_.fs()<<std::fixed<<std::setw(10)<<energy_error_bar_.mean();
  optimal_energy_.fs()<<std::setw(10)<<num_vmc_samples_; 
  if (status == exit_status::converged) {
    optimal_energy_.fs()<<std::setw(11)<<"CONVERGED"<<std::setw(7)<<0<<std::endl;
  }
  else {
    optimal_energy_.fs()<<std::setw(11)<<"NOT_CONVD"<<std::setw(7)<<0<<std::endl;
  }
  // save the values
  optimal_energy_.save_result();
  energy_error_bar_.save_result();
  // grand average for samples done so far
  optimal_energy_.fs()<<std::right;
  optimal_energy_.fs()<<std::scientific<<std::uppercase<<std::setprecision(6);
  optimal_energy_.fs() << "#" << std::string(72, '-') << "\n";

  optimal_energy_.fs() << "# grand average:\n"; // 
  optimal_energy_.fs() << "#" << std::string(72, '-') << "\n";
  for (const auto& p : xvar_values_) 
    optimal_energy_.fs()<<std::setw(14)<<p;
  optimal_energy_.fs()<<std::setw(15)<<optimal_energy_.grand_data().mean();
  optimal_energy_.fs()<<std::fixed<<std::setw(10)<<energy_error_bar_.grand_data().mean();
  optimal_energy_.fs()<<std::setw(10)<<num_vmc_samples_; 
  if (all_converged_) 
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
  optimal_parms_.save_result();
  // grand average for samples done so far
  optimal_parms_.print_grand_result(xvar_values_);

  // reset for the next sample
  optimal_energy_.reset();
  energy_error_bar_.reset();
  optimal_parms_.reset();
  return 0;
}

std::ostream& Optimizer::print_progress(std::ostream& os, 
  const int& iter, const std::string& algorithm)
{
  std::ios state(NULL);
  state.copyfmt(os);
  os << "#" << std::string(60, '-') << std::endl;
  os << std::left;
  os << " iteration   =  "<<std::setw(6)<<iter<<"\n";
  os << " step        =  "<< algorithm <<"\n";
  os.copyfmt(state);
  return os;
}

std::ostream& Optimizer::print_progress(std::ostream& os, 
  const var::parm_vector& vparms, const double& energy,
  const double& error_bar, const RealVector& grad, const RealVector& search_dir,
  const double& gnorm, const double& avg_gnorm, const double& en_trend)
{
  std::ios state(NULL);
  state.copyfmt(os);
  os <<std::scientific<<std::uppercase<<std::setprecision(6)<<std::right;
  os << " energy      ="<<std::setw(14)<<energy << "   (+/-) ";
  os <<std::fixed<<std::setw(10)<<error_bar<<"\n";
  os <<std::scientific<<std::right;
  os <<std::scientific<<std::uppercase<<std::setprecision(4)<<std::right;
  os << " varp        =";
  for (int i=0; i<num_parms_print_; ++i) os<<std::setw(12)<<vparms[i];
  os << "\n";
  os << " grad        =";
  for (int i=0; i<num_parms_print_; ++i) os<<std::setw(12)<<grad[i];
  os << "\n";
  os << " search_dir  =";
  for (int i=0; i<num_parms_print_; ++i) os<<std::setw(12)<<search_dir[i];
  os << "\n";
  os << " gnorm       ="<<std::setw(12)<< gnorm;
  os << " (avg ="<<std::setw(11)<<avg_gnorm<<")\n";
  os <<std::fixed<<std::setprecision(6)<<std::right;
  os <<" MK trend    = "<<std::setw(9)<<en_trend<<"  ";
  //os <<std::setw(9)<<mk_trend;
  //os << " ("<<trend_elem<<")"<<"\n"; 
  os << "\n"; 
  os.copyfmt(state);
  return os;
}


/*
int Optimizer::genetic_algorthim(const RealVector& grad, RealMatrix& srmat, 
  RealMatrix& srmat_inv, RealVector& search_dir)
{
  while (not converged) {
    vmc.run(vparms,en,en_err,true);
  }
}
*/


int Optimizer::stochastic_reconf(const RealVector& grad, RealMatrix& srmat, 
  RealMatrix& srmat_inv, RealVector& search_dir)
{
  Eigen::BDCSVD<Eigen::MatrixXd> bdcsvd(num_parms_,num_parms_);  

  /*---------------------------------------------------- 
    Conditioning of the SR matrix
  *-----------------------------------------------------*/
  for (int i=0; i<num_parms_; ++i) {
    srmat(i,i) *= (1.0+sr_diag_scale_);
    srmat(i,i) += sr_diag_shift_;
  }

  /*---------------------------------------------------- 
    Solve the equation: [S][d] = -g, S is SR matrix, 
    d is search dir. g is gradient.
  *-----------------------------------------------------*/
  if (SR_solver_==SR_solver::BDCSVD) {
    search_dir = -BDCSVD_.compute(srmat, Eigen::ComputeThinU | Eigen::ComputeThinV).solve(grad);
  }

  else if (SR_solver_==SR_solver::JacobiSVD) {
    JacobiSVD_.compute(srmat, Eigen::ComputeThinU | Eigen::ComputeThinV);

    // Cut-Off redundant direction (using SVD decomposition)
    double lmax = JacobiSVD_.singularValues()(0);
    int m_cut = num_parms_;
    RealVector svals = JacobiSVD_.singularValues();
    for (int m=0; m<num_parms_; ++m) {
      if (svals[m]/lmax < w_svd_cut_) {
        m_cut = m;
        break;
      }
    }
    for (int i=0; i<num_parms_; ++i) {
      for (int j=0; j<num_parms_; ++j) {
        double sum = 0.0;
        for (int m=0; m<m_cut; ++m) {
          sum += JacobiSVD_.matrixV()(i,m)*JacobiSVD_.matrixU()(j,m)/svals[m];
        }
        srmat_inv(i,j) = sum;
        //std::cout << "srmat_inv["<<i<<","<<j<<"] = " << sum << "\n";
      }
    }
    //getchar();
    search_dir = -srmat_inv*grad;
  }

  else if (SR_solver_==SR_solver::EigSolver) {
    EigSolver_.compute(srmat);

    // determine m_cut (since eigenvalues are in ascending order)
    double lmax = EigSolver_.eigenvalues()(num_parms_-1);
    int m_cut = 0;
    for (int m=0; m<num_parms_; ++m) {
      if (EigSolver_.eigenvalues()[m]/lmax >= w_svd_cut_) {
        m_cut = m; break;
      }
    }
    for (int i=0; i<num_parms_; ++i) {
      for (int j=0; j<num_parms_; ++j) {
        double sum = 0.0;
        for (int m=m_cut; m<num_parms_; ++m) {
          sum += EigSolver_.eigenvectors()(m,i)*EigSolver_.eigenvectors()(m,j)/EigSolver_.eigenvalues()(m);
        }
        srmat_inv(i,j) = sum;
      }
    }
    // update search direction
    search_dir = -srmat_inv*grad;
  }
  else {
    throw std::range_error("Optimizer::stochastic_reconf: unknown SR solver");
  }

  return 0;
}

bool Optimizer::backtracking_Armjio_step(VMCRun& vmc, RealVector& vparms, double& en, 
    double& en_err, RealVector& grad, RealVector& grad_err, 
    const RealVector& search_dir)
{
  double tau = 0.8;
  double alpha = CG_alpha0_;
  double Armjio_rho = 1.0E-2;
  double gtp_factor = Armjio_rho * grad.dot(search_dir);

  double f0 = en;
  double f0_err = en_err;
  RealVector x0 = vparms;
  RealVector g0 = grad;

  if (print_progress_) std::cout << " line search =  Armjio backtracting: ";
  if (print_log_) logfile_ << " line search =  Armjio backtracting: ";
  for (int iter=0; iter<max_ls_steps_; ++iter) {
    if (print_progress_) {
      std::cout << iter+1 << ".." << std::flush;
    }
    if (print_log_) {
      logfile_ << iter+1 << ".."; //<< std::flush; 
    }
    // update the parameters
    vparms.noalias() = x0 + alpha*search_dir;
    apply_projection(vparms);
    vmc.run(vparms,en,en_err,grad,grad_err,true);
    if (en+en_err <= f0 + alpha*gtp_factor) {
      if (print_progress_) {
        std::cout<<" done\n";
        std::cout<<" step size   =  "<<alpha<<"\n";
      } 
      if (print_log_) {
        logfile_<<" done\n";
        logfile_<<" step size   =  "<<alpha<<"\n";
      }
      return true;
    }
    alpha *= tau;
  }
  // line search failed
  if (print_progress_) {
    std::cout<<" aborted\n";
  } 
  if (print_log_) {
    logfile_<<" aborted\n";
  }
  en = f0;
  en_err = f0_err;
  vparms = x0;
  grad = g0;

  return false;
}

double Optimizer::line_search(VMCRun& vmc, RealVector& vparms, 
  double& en, double& en_err, RealVector& grad, RealVector& grad_err, 
  const RealVector& search_dir)
{
  if (print_progress_) std::cout << " line search =  ";
  if (print_log_) logfile_ << " line search =  "; 
  std::string message;
  bool search_done = false;
  int search_step = 0;
  double net_dt = 0.0;
  // start LS
  double dt = probLS_.start(en,en_err,grad,grad_err,search_dir);
  net_dt += dt;

  // do steps
  while (true) {
    search_step++;
    if (print_progress_) {
      std::cout << search_step << ".." << std::flush;
    }
    if (print_log_) {
      logfile_ << search_step << ".."; //<< std::flush; 
    }

    // run with next parameters
    vparms += dt*search_dir;
    vparms = varp_lbound_.cwiseMax(vparms.cwiseMin(varp_ubound_));
    vmc.run(vparms,en,en_err,grad,grad_err,true);

    // find next step size by probabilistic line search method
    dt = probLS_.do_step(en,en_err,grad,grad_err,search_dir,search_done,message);
    net_dt += dt;

    // check if done
    if (search_done) {
      if (print_progress_) {
        //std::cout<<" done\n";
        //std::cout<<" step size   =  "<<net_dt<<" ("<<message<<")\n";
        std::cout<<" "<<message<<"\n";
        std::cout<<" step size   =  "<<net_dt<<"\n";
      } 
      if (print_log_) {
        logfile_<<" done\n";
        logfile_<<" step size   =  "<<net_dt<<" ("<<message<<")\n";
      }
      break;
    } 
    //std::cout << "dt = " << dt << "\n";
  }
  // return step size
  return net_dt;
}

void Optimizer::apply_projection(RealVector& vparms)
{
  if (!varp_bound_exists_) return;
  for (int n=0; n<num_parms_; ++n) {
    if (varp_bounded_[n]) {
      if (vparms[n]<varp_lbound_[n]) vparms[n] = varp_lbound_[n];
      if (vparms[n]>varp_ubound_[n]) vparms[n] = varp_ubound_[n];
    }
  }
}

double Optimizer::squaredNormProj(const RealVector& grad, const RealVector& vparms) const
{
  if (!varp_bound_exists_) return grad.squaredNorm();
  double gnorm = 0.0;
  for (int n=0; n<num_parms_; ++n) {
    double gn = grad[n];
    if (varp_bounded_[n]) {
      if (vparms[n]>=varp_lbound_[n] && vparms[n]<=varp_ubound_[n]) {
        gnorm += gn*gn;
      }
    }
    else {
      gnorm += gn*gn;
    }
  }
  return gnorm;
} 

double Optimizer::projected_gnorm(const RealVector& x, const RealVector& grad, 
  const RealVector& lb, const RealVector& ub) const
{
  const int n = x.size();
  double res = 0.0;
  for (int i=0; i<n; ++i) {
    double proj = std::max(lb[i], x[i] - grad[i]);
    proj = std::min(ub[i], proj);
    res = std::max(res, std::abs(proj - x[i]));
  }
  return res;
}

double Optimizer::series_avg(const std::deque<double>& data) const
{
  if (data.size() == 0) return 0.0;
  double sum = 0.0;
  for (const auto& d : data) sum += d;
  return sum/data.size();
}

int Optimizer::stochastic_CG(const RealVector& grad, const RealVector& grad_xprev, 
  RealVector& stochastic_grad, RealVector& search_dir)
{
  // Norm of previous stochastic gradient
  double gnorm_prev = stochastic_grad.squaredNorm();

  // Update stocahastic gradient (SARAH algorithm)
  RealVector grad_diff = grad - grad_xprev;
  stochastic_grad += grad_diff;

  // Conjugate search direction
  // compute beta
  double beta;
  if (CG_Algorithm_==CG_type::FR) {
    beta = stochastic_grad.squaredNorm()/gnorm_prev;
  }
  else if (CG_Algorithm_==CG_type::PR) {
    beta = stochastic_grad.dot(grad_diff)/gnorm_prev;
  }
  else if (CG_Algorithm_==CG_type::HS) {
    beta = stochastic_grad.dot(grad_diff)/search_dir.dot(grad_diff);
  }
  else if (CG_Algorithm_==CG_type::DY) {
    beta = stochastic_grad.squaredNorm()/search_dir.dot(grad_diff);
  }
  else {
    throw std::range_error("Optimizer::stochastic_CG: unknown CG method");
  }
  //std::cout << "beta = " << beta << "\n"; //getchar();

  // new search direction
  search_dir = -stochastic_grad+beta*search_dir;

  return 0;
}


} // end namespace vmc













