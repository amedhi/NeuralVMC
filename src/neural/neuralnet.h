/*
* @Author: Amal Medhi
* @Date:   2018-12-29 20:17:54
* @Last Modified by:   amedhi
* @Last Modified time: 2019-01-08 23:26:32
*----------------------------------------------------------------------------*/
#ifndef NEURALNET_H
#define NEURALNET_H

#include <vector>
#include <memory>
#include "layer.h"

namespace nnet {

class SequentialNet: private std::vector<NeuralLayer>
{
public:
  SequentialNet() : num_layers_{0} { clear(); pid_range_.clear(); } 
  ~SequentialNet() {}
  int add_layer(const int& units, const std::string& activation="None", 
    const int& input_dim=0);
  void compile(void);
  //void set_input(const Vector& input) { front().set_input(input); }
  const int& num_params(void) const { return num_params_; }
  const double& get_parameter(const int& id) const;
  void get_parameters(Vector& pvec) const;
  void update_parameters(const Vector& pvec);
  void update_parameter(const int& id, const double& value);
  void run(const eig::real_vec& input); 
  eig::real_vec get_output(const eig::real_vec& input) const;
  const Vector& output(void) const { return output_; }
  const Matrix& get_gradient(void);
private:
  int num_layers_{0};
  int num_params_{0};
  std::vector<int> pid_range_;
  Vector output_;
  Matrix gradient_;
};

/*
class NeuralNet  
{
public:
  NeuralNet() { layers_.clear(); }
  ~NeuralNet() {}
  int add_layer(const int& units, const std::string& activation="None", 
  const int& input_dim=0);
  Vector get_output(const Vector& input);
//void add_layer(const Layer& layer); 
private:
  std::vector<Layer> layers_;
};
*/


} // end namespace nnet
#endif