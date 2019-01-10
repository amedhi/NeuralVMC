/*---------------------------------------------------------------------------
* @Author: Amal Medhi
* @Date:   2018-12-29 10:48:33
* @Last Modified by:   amedhi
* @Last Modified time: 2019-01-08 23:27:00
*----------------------------------------------------------------------------*/
#ifndef LAYER_H
#define LAYER_H

#include <iostream>
#include <string>
#include <memory>
#include <Eigen/Core>
#include "activation.h"

namespace nnet {

class NeuralLayer
{
public:
  virtual ~NeuralLayer() {}
  virtual void set_kernel(const Matrix& w) = 0; 
  virtual void set_bias(const Vector& b) = 0;
  virtual void set_input(const Vector& v) { input_=v; }
  virtual void set_input_layer(NeuralLayer* inlayer) = 0;
  virtual void set_output_layer(NeuralLayer* outlayer) = 0;
  virtual const Matrix& get_kernel(void) const { return kernel_; } 
  virtual const Vector& get_bias(void) const { return bias_; }
  virtual const Vector& get_input(void) const { return input_; }
  virtual Vector get_output(void) const = 0; 
  virtual const int& num_units(void) const { return num_units_; }
protected:
  static int num_layers_;
  int id_{0};
  int num_units_{1};
  Vector input_;
  Matrix kernel_;
  Vector bias_;
};

class InputLayer : public NeuralLayer
{
public:
  InputLayer(const int& units=1); 
  ~InputLayer() { num_layers_--; }
  void set_kernel(const Matrix& w) override {}
  void set_bias(const Vector& b) override {}
  void set_input_layer(NeuralLayer* layer) override {}
  void set_output_layer(NeuralLayer* layer) override { outlayer_=layer; }
  Vector get_output(void) const override { return input_; } 
private:
  NeuralLayer* outlayer_{nullptr};
};

class DenseLayer : public NeuralLayer
{
public:
  DenseLayer(const int& units, const std::string& activation="None", 
    const int& input_dim=1);
  ~DenseLayer() { num_layers_--; }
  void set_kernel(const Matrix& w) override {}
  void set_bias(const Vector& b) override {}
  void set_input_layer(NeuralLayer* layer) override { inlayer_=layer; }
  void set_output_layer(NeuralLayer* layer) override { outlayer_=layer; }
  Vector get_output(void) const override;
private:
  int input_dim_{1};
  std::shared_ptr<Activation> activation_{nullptr};
  NeuralLayer* inlayer_{nullptr};
  NeuralLayer* outlayer_{nullptr};
};


/*
class Layer
{
public:
  //Layer() { id_=num_layers_++; }
  Layer(const int& units, const std::string& activation="None", 
  	const int& input_dim=1);
  //Layer(const Layer& layer);
  ~Layer() { --num_layers_; } // inlayer_.reset(nullptr); outlayer_.reset(nullptr); }
  void set_input_dim(const int& input_dim) 
  { 
  	input_dim_=input_dim; 
  	kernel_=Matrix::Ones(num_units_,input_dim_);
  	input_=Vector::Zero(input_dim_);
  }
  void set_input(const Vector& input) { input_=input; } 
  void set_input_layer(Layer* inlayer);
  void set_output_layer(Layer* outlayer);
  void set_kernel(const Matrix& w) { kernel_=w; }
  void set_bias(const Vector& b) { bias_=b; }
  const int& num_units(void) const { return num_units_; }
  Vector get_output(void) const; 
  const Vector& get_input(void) const { return input_; } 
  std::string name_;
  Layer* inlayer_{nullptr};
  Layer* outlayer_{nullptr};
private:
  static int num_layers_;
  int id_{0};
  int num_units_{1};
  int input_dim_{1};
  std::shared_ptr<Activation> activation_{nullptr};
  Matrix kernel_;
  Vector bias_;
  Vector input_;
};
*/







} // end namespace nnet


#endif