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

class SequentialNet: private std::vector<std::unique_ptr<NeuralLayer>>
{
public:
  SequentialNet() { clear(); } 
  ~SequentialNet() {}
  int add_layer(const int& units, const std::string& activation="None", 
    const int& input_dim=0);
  Vector get_output(const Vector& input);
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