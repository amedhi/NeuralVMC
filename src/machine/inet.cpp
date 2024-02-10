/*---------------------------------------------------------------------------
* @Author: Amal Medhi
* @Date:   2018-12-29 20:39:14
* @Last Modified by:   Amal Medhi
* @Last Modified time: 2024-02-10 19:04:10
*----------------------------------------------------------------------------*/
#include <boost/filesystem.hpp>
#include <boost/tokenizer.hpp>
#include <filesystem>
#include "inet.h"

namespace ann {

INet::INet() : AbstractNet(), num_layers_{0}, num_params_{0}
{
  num_layers_ = 0;
  num_output_units_ = 1;
  num_params_ = 0;
  output_.resize(1);
}

int INet::add_layer(const int& units, const std::string& activation, 
  const int& input_dim)
{
  throw std::invalid_argument("INet::add_layer: invalid call");
}

int INet::add_sign_layer(const int& input_dim)
{
  throw std::invalid_argument("INet::add_sign_layer: invalid call");
}

void INet::init_parameters(random_engine& rng, const double& sigma) 
{
  // nothing to be done
}

void INet::init_parameter_file(const std::string& save_path, const std::string& load_path)
{
  // nothing to be done
}

void INet::save_parameters(void) const
{
  // nothing to be done
}

void INet::load_parameters(void)
{
  // nothing to be done
}

const double& INet::get_parameter(const int& id) const
{
  throw std::out_of_range("INet::get_parameter: no parameter exists");
}

void INet::get_parameters(RealVector& pvec) const
{
  // nothing to be done
}

void INet::get_parameter_names(std::vector<std::string>& pnames, const int& pos) const
{
  // nothing to be done
}

void INet::get_parameter_values(RealVector& pvalues, const int& pos) const
{
  // nothing to be done
}

void INet::update_parameters(const RealVector& pvec, const int& start_pos)
{
  // nothing to be done
}

void INet::update_parameter(const int& id, const double& value)
{
  // nothing to be done
}

void INet::do_update_run(const RealVector& input)
{
  // nothing to be done
  output_[0] = 1.0;
}

void INet::do_update_run(const RealVector& new_input, const std::vector<int> new_elems) 
{
  // nothing to be done
  output_[0] = 1.0;
}

RealVector INet::get_new_output(const RealVector& input) const
{
  // nothing to be done
  return RealVector::Constant(1,1.0);
}

RealVector INet::get_new_output(const RealVector& new_input, const std::vector<int> new_elems) const
{
  // nothing to be done
  return RealVector::Constant(1,1.0);
}

void INet::get_gradient(RealMatrix& grad) const
{
  // nothing to be done
}

void INet::get_log_gradient(RealMatrix& grad) const 
{
  // nothing to be done
}



} // end namespace ann

