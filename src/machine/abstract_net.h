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
#include "dtype.h"

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
  virtual void init_parameter_file(const std::string& prefix) = 0;
  virtual void init_parameters(random_engine& rng, const double& sigma) = 0;
  virtual void save_parameters(void) const = 0;
  virtual void load_parameters(const std::string& load_path) = 0;
  virtual void get_parameter_names(std::vector<std::string>& pnames, 
    const int& pos=0) const = 0;
  virtual void get_parameter_values(Vector& pvalues, 
    const int& pos=0) const = 0;
  virtual void get_parameter_vector(std::vector<double>& pvalues, 
    const int& pos) const = 0;
  virtual void update_parameters(const Vector& pvec, const int& pos=0) = 0;
  virtual void update_parameter(const int& id, const double& value) = 0;
  virtual void do_update_run(const eig::real_vec& input) = 0; 
  virtual void do_update_run(const Vector& new_input, const std::vector<int> new_elems) = 0; 
  virtual const Vector& output(void) const = 0;
  virtual Vector get_new_output(const Vector& input) const = 0;
  virtual Vector get_new_output(const Vector& new_input, const std::vector<int> new_elems) const = 0; 
  virtual const Matrix& get_gradient(void) const = 0;
};


} // end namespace ann

#endif