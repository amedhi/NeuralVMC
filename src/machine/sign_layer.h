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
  int update_forward(const RealVector& input) override;
  int update_forward(const RealVector& new_input, const std::vector<int>& new_elems, 
    const RealVector& input_changes) override; 
  int feed_forward(const RealVector& input) const override; 
  int feed_forward(const RealVector& new_input, const std::vector<int>& new_elems, 
  	const RealVector& input_changes) const override;
  RealVector get_new_output(const RealVector& input) const override; 
  int derivative(RealMatrix& derivative, const int& num_total_params) const override;
  //int back_propagate(const int& pid_end, const RowVector& backflow,
  //  Matrix& derivative, const int& use_col) const override;
private:
	double re_phase_;
	double im_phase_;
	RealVector cos_thetak_;
	RealVector sin_thetak_;
};


} // end namespace ann


#endif