/*---------------------------------------------------------------------------
* @Author: amedhi
* @Date:   2019-01-10 14:22:50
* @Last Modified by:   Amal Medhi
* @Last Modified time: 2022-07-11 16:59:17
*----------------------------------------------------------------------------*/
#include "activation.h"

namespace ann {

RELU::RELU(const double& alpha, const double& threshold, const double& maxval) 
  	: alpha_{alpha}, threshold_{threshold}, maxval_{maxval}
{
	if (alpha_>=0.0 || threshold_>=0.0 || maxval_>=0) {
		default_ = false;
		if (alpha_<0.0 || threshold_<0.0 || maxval_<0) {
      throw std::invalid_argument("RELU::RELU: invalid argument value(s)");
		}
		if (threshold_>=maxval_) {
      throw std::invalid_argument("RELU::RELU: invalid argument value(s)");
		}
	}
	else default_ = true;
}

double RELU::function(const double& x) const
{
	if (default_) {
		return std::max(x,0.0);
	}
	else {
		if (x>=maxval_) return maxval_;
		else if (x>=threshold_) return x;
		else return alpha_ * (x-threshold_);
	}
}

Vector RELU::function(const Vector& input) const
{
	if (default_) {
		return input.unaryExpr([](const double& x) { return std::max(x,0.0); });
	}
	else {
		return input.unaryExpr([this](const double& x) 
			{ 
				if (x>=maxval_) return maxval_;
				else if (x>=threshold_) return x;
				else return alpha_ * (x-threshold_);
			} 
		);
	}
}

double RELU::derivative(const double& x) const
{
	if (default_) {
		return x >= 0.0? 1.0 : 0.0;
	}
	else {
		if (x>=maxval_) return 0.0;
		else if (x>=threshold_) return 1.0;
		else return alpha_;
	}
}

Vector RELU::derivative(const Vector& input) const
{
	if (default_) {
		return input.unaryExpr([](const double& x) { return x>=0.0?1.0:0.0; });
	}
	else {
		return input.unaryExpr([this](const double& x) 
			{ 
				if (x>=maxval_) return 0.0;
				else if (x>=threshold_) return 1.0;
				else return alpha_;
			} 
		);
	}
}

double TANH::function(const double& x) const 
{
  return std::tanh(x);
}

Vector TANH::function(const Vector& input) const
{
  return input.unaryExpr([](const double& x) {return std::tanh(x);});
}

double TANH::derivative(const double& x) const 
{
  double y = std::tanh(x);
  return 1.0-y*y;
}

Vector TANH::derivative(const Vector& input) const
{
  return input.unaryExpr([](const double& x) 
	{ double y = std::tanh(x); return 1.0-y*y; });
}

double Sigmoid::function(const double& x) const 
{
  return 1.0/(1.0+std::exp(-x));
}

Vector Sigmoid::function(const Vector& input) const
{
  return input.unaryExpr([](const double& x) {return 1.0/(1.0+std::exp(-x));});
}

double Sigmoid::derivative(const double& x) const 
{
  double y = 1.0/(1.0+std::exp(-x));
  return y*(1.0-y);
}

Vector Sigmoid::derivative(const Vector& input) const
{
  return input.unaryExpr([](const double& x) 
	{ double y = 1.0/(1.0+std::exp(-x)); return y*(1.0-y); });
}

double ShiftedSigmoid::function(const double& x) const 
{
  return 1.0/(1.0+std::exp(-x))-0.5;
}

Vector ShiftedSigmoid::function(const Vector& input) const
{
  return input.unaryExpr([](const double& x) {return 1.0/(1.0+std::exp(-x))-0.5;});
}

double ShiftedSigmoid::derivative(const double& x) const 
{
  double y = 1.0/(1.0+std::exp(-x));
  return y*(1.0-y);
}

Vector ShiftedSigmoid::derivative(const Vector& input) const
{
  return input.unaryExpr([](const double& x) 
	{ double y = 1.0/(1.0+std::exp(-x)); return y*(1.0-y); });
}


//---------------- log[cosh(x)]-----------------------
double LCOSH::function(const double& x) const 
{
  return std::log(std::cosh(x));
}

Vector LCOSH::function(const Vector& input) const
{
  return input.unaryExpr([](const double& x) {return std::log(std::cosh(x));});
}

double LCOSH::derivative(const double& x) const 
{
  return std::tanh(x);
}

Vector LCOSH::derivative(const Vector& input) const
{
  return input.unaryExpr([](const double& x) 
	{ return std::tanh(x); });
}

//---------------- cos(pi*x)-----------------------
double COSPI::function(const double& x) const 
{
  //return std::cos(pi()*x);
  return std::cos(M_PI*x);
}

Vector COSPI::function(const Vector& input) const
{
  //return input.unaryExpr([](const double& x) {return std::cos(pi()*x);});
  return input.unaryExpr([](const double& x) {return std::cos(M_PI*x);});
}

double COSPI::derivative(const double& x) const 
{
  //return -pi()*std::sin(pi()*x);
  return -M_PI*std::sin(M_PI*x);
}

Vector COSPI::derivative(const Vector& input) const
{
  //return input.unaryExpr([](const double& x) 
	//{ return -pi()*std::sin(pi()*x); });
  return input.unaryExpr([](const double& x) 
	{ return -M_PI*std::sin(M_PI*x); });
}






} // end namespace ann