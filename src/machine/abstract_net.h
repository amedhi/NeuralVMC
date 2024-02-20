/*---------------------------------------------------------------------------
* Author: Amal Medhi
* Date:   2018-27-12 13:19:36
* Last Modified by:   Amal Medhi, amedhi@macbook
* Last Modified time: 
* Copyright (C) Amal Medhi, amedhi@iisertvm.ac.in
*----------------------------------------------------------------------------*/
#ifndef ABSTRACT_NET_H
#define ABSTRACT_NET_H

#include <iostream>
#include <string>
#include <limits>
#include "../matrix/matrix.h"
#include "./dtype.h"

namespace ann {

class AbstractNet
{
public:
	//AbstractNet() {}
	virtual ~AbstractNet() {}
  virtual int add_layer(const int& units, const std::string& activation="None", 
    const int& input_dim=0) = 0;
  virtual int add_sign_layer(const int& input_dim) = 0;
  virtual int compile(void) = 0;
  virtual const int& num_params(void) const = 0;
  virtual const int& num_output_units(void) const = 0;
  virtual void init_parameter_file(const std::string& save_path, const std::string& load_path) = 0;
  virtual void init_parameters(random_engine& rng, const double& sigma) = 0;
  virtual void save_parameters(void) const = 0;
  virtual void load_parameters(void) = 0;
  virtual void get_parameter_names(std::vector<std::string>& pnames, 
    const int& pos=0) const = 0;
  virtual void get_parameter_values(RealVector& pvalues, 
    const int& pos=0) const = 0;
  virtual void get_parameter_lbound(RealVector& lbound, const int& pos=0) const = 0;
  virtual void get_parameter_ubound(RealVector& ubound, const int& pos=0) const = 0;
  virtual void update_parameters(const RealVector& pvec, const int& pos=0) = 0;
  virtual void update_parameter(const int& id, const double& value) = 0;
  virtual void do_update_run(const RealVector& input) = 0; 
  virtual void do_update_run(const RealVector& new_input, const std::vector<int> new_elems) = 0; 
  virtual const RealVector& output(void) const = 0;
  virtual RealVector get_new_output(const RealVector& input) const = 0;
  virtual RealVector get_new_output(const RealVector& new_input, const std::vector<int> new_elems) const = 0; 
  virtual double get_new_output_ratio(const RealVector& input) const = 0;
  virtual double get_new_output_ratio(const RealVector& new_input, const std::vector<int> new_elems) const = 0; 
  // gradient in case multi-output network
  virtual void get_gradient(RealMatrix& grad_mat) const = 0;
  virtual void get_log_gradient(RealMatrix& grad_mat) const = 0;

protected:
  // parameter file
  std::string save_path_{""};
  std::string load_path_{""};
  std::string fname_{""};
};


} // end namespace ann

#endif