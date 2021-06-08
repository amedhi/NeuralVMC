/*---------------------------------------------------------------------------
* @Author: Amal Medhi
* @Date:   2018-12-29 10:48:33
* @Last Modified by:   amedhi
* @Last Modified time: 2019-01-08 23:27:00
*----------------------------------------------------------------------------*/
#ifndef SIGN_LAYER_H
#define SIGN_LAYER_H

#include "neural_layer.h"

namespace ann {

class SignLayer : public NeuralLayer
{
public:
  SignLayer(const std::string& activation="None", const int& input_dim=1);
  ~SignLayer() {}
  void set_output_layer(NeuralLayer* layer) override;
  int update_forward(const Vector& input) override;
  int update_forward(const Vector& new_input, const std::vector<int>& new_elems, 
    const Vector& input_changes) override; 
  int feed_forward(const Vector& input) const override; 
  int feed_forward(const Vector& new_input, const std::vector<int>& new_elems, 
  	const Vector& input_changes) const override;
  Vector get_new_output(const Vector& input) const override; 
  int derivative(Matrix& derivative, const int& num_total_params) const override;
  //int back_propagate(const int& pid_end, const RowVector& backflow,
  //  Matrix& derivative, const int& use_col) const override;
private:
	double re_phase_;
	double im_phase_;
	Vector cos_thetak_;
	Vector sin_thetak_;
};


} // end namespace ann


#endif