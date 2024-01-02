/*---------------------------------------------------------------------------
* @Author: Amal Medhi
* @Date:   2018-12-29 20:39:14
* @Last Modified by:   Amal Medhi
* @Last Modified time: 2024-01-02 13:08:55
*----------------------------------------------------------------------------*/
#include <boost/filesystem.hpp>
#include <boost/tokenizer.hpp>
#include <filesystem>
#include "rbm.h"

namespace ann {

RBM::RBM() : AbstractNet(), num_layers_{0}, num_params_{0}
{
  layers_.clear();
  num_params_fwd_.clear(); 
}

RBM::RBM(const lattice::Lattice& lattice, const input::Parameters& inputs) 
  : AbstractNet(), num_layers_{0}, num_params_{0}
{
  construct(lattice, inputs);
}

int RBM::construct(const lattice::Lattice& lattice, const input::Parameters& inputs)
{
  layers_.clear();
  num_params_fwd_.clear(); 
  /*
   * Network parameters = # of Kernel params + # of hidden layer bias params.
   *---------------------------------------------------------------
   * In the unsymmetrized case, the no of kernel (w) parameters is 
   * = No of hidden units * No of visible units
   * = [(No of visible units) * (alpha)] * (no of visible units) 
   * = [(2*num_sites) * alpha] * (2*num_sites)
   * 
   * Symmetries:
   * Assuming 'spin flip' symmetry, the w-parameters for spin-UP and spin-DN
   * units are identical. So total no of w-parameters reduces to
   * = [(2*num_sites) * alpha] * (num_sites)
   *
   * Translational symmetry:
   * The parameter 'alpha' is constrained to have only integer values.
   * Let's take 'alpha=1'. The first 'num_basis_sites' rows contain 
   * = (num_basis_sites*num_sites) distinct parameters. Other rows 
   * parmaters are obtained by appliying the translational symmetry
   * operations to these row elements.
   * For alpha>1, the number distict w-parameters is
   * = alpha * num_basis_sites * num_sites
   * 
   * Therefore, total number of parameters is:
   * = alpha * num_basis_sites * num_sites + 1
   * The last parameter is the hidden layer bias parameters.
  */
  symmetry_ = true;

  // network structure assuming 'Lattice translational' & 'Spin-flip' symmetry
  num_sites_ = lattice.num_sites();
  num_basis_sites_ = lattice.num_basis_sites();
  num_visible_units_ = 2*num_sites_;
  num_hblocks_ = 1;
  num_hidden_units_ = num_visible_units_*num_hblocks_;
  vbias_ = Vector::Random(num_visible_units_);
  hbias_ = Vector::Random(num_hidden_units_);
  kernel_ = Matrix::Random(num_hidden_units_,num_visible_units_);
  input_.resize(num_visible_units_);
  lin_output_.resize(num_hidden_units_);
  cosh_output_.resize(num_hidden_units_);
  output_.resize(1);
  // no of parameters
  num_kernel_params_ = (2*num_basis_sites_*num_hblocks_)*num_sites_;
  num_hbias_params_ = 2*num_basis_sites_*num_hblocks_;
  num_params_ =  num_kernel_params_ + num_hbias_params_;
  pvector_.resize(num_params_);
  gradient_.resize(num_params_,1);

  /*
   * Symmetry Map:
   *---------------------------------------------------------------
   * The Group of translational symmetries G={T_R} has 'L' number of elements 
   * (including the identity element), where L = number of unit cells
   * Let's say that the k-th symmetry opetation map a site with index 'm' 
   * to another site with index 'n';
   * In the following, we store this mapping as, 'tsymm_map[k,m] = n'
   */
  num_tsymms_ = lattice.num_unitcells();
  tsymm_map_.resize(num_tsymms_,num_sites_);
  for (const auto& s : lattice.sites()) {
    int m = s.id();
    Vector3i bravidx = Vector3i(0,0,0);
    for (int k=0; k<num_tsymms_; ++k) {
      auto ts = lattice.translated_site(s, bravidx);
      tsymm_map_(k,m) = ts.id();
      //std::cout << "TR["<<k<<","<<m<<"] = "<<tsymm_map_(k,m)<<"\n"; getchar();
      bravidx = lattice.get_next_bravindex(bravidx);
    }
  }

  /*
   * Parameter Map:
   *---------------------------------------------------------------
   * A parameter appear in more than one location in the Kernel matrix.
   * Store the locations where each parameter appear.
   */
  kernel_params_map_.clear();
  kernel_params_map_.resize(num_kernel_params_);
  // set unique sequential value to the parameters
  for (int i=0; i<num_kernel_params_; ++i) pvector_[i] = i;
  update_kernel_params(pvector_(Eigen::seqN(0,num_kernel_params_)));
  for (int i=0; i<kernel_.rows(); ++i) {
    for (int j=0; j<kernel_.cols(); ++j) {
      int n = std::nearbyint(kernel_(i,j));
      kernel_params_map_[n].push_back({i,j});
    }
  }
  
  /*
  std::cout << "kernel_params_map\n";
  for (int i=0; i<num_kernel_params_; ++i) {
    for (const auto& elem: kernel_params_map_[i]) {
      std::cout<<"("<<elem.first<<","<<elem.second<<")  ";
    }
    std::cout << "\n";
  }*/

  // bias parameters map
  bias_params_map_.clear();
  bias_params_map_.resize(num_hbias_params_);
  for (int i=0; i<num_hbias_params_; ++i) pvector_[i] = i;
  update_hbias_params(pvector_(Eigen::seqN(0,num_hbias_params_)));
  for (int i=0; i<hbias_.size(); ++i) {
    int n = std::nearbyint(hbias_(i));
    bias_params_map_[n].push_back(i);
  }
  /*
  for (int i=0; i<num_hbias_params_; ++i) {
    for (const auto& elem: bias_params_map_[i]) {
      std::cout<<elem<<"  ";
    }
    std::cout << "\n";
  }*/


  // Initialize the parameters
  pvector_.setRandom(); 
  // kernel parameters
  update_kernel_params(pvector_(Eigen::seqN(0,num_kernel_params_)));
  // bias
  update_hbias_params(pvector_.tail(num_hbias_params_));

  //std::cout << "Exiting at RBM::construct\n";
  //exit(0);
  return 0;
}

int RBM::update_kernel_params(const RealVector& params)
{
  assert(params.size()==num_kernel_params_);
  /* 
   * First, update the first-half of the columns. The second-half is equal
   * to the first-half by 'spin-flip' symmetry.
   */
  int N = num_basis_sites_;
  int L = num_sites_;
  int M = N*L;
  Matrix mat(N,L);
  int block_row = 0;
  int ppos = 0;
  for (int block=0; block<num_hblocks_; ++block) {
    int row = 0;
    // Upper (spin-up) half
    mat = params(Eigen::seq(ppos,ppos+M-1)).reshaped(N,L);
    // translate & populate the kernel
    for (int T=0; T<num_tsymms_; ++T) {
      kernel_.block(block_row+row,0,N,L) = row_translate(mat, T);
      row += N;
    }
    // Lower (spin-dn) half
    mat = params(Eigen::seq(ppos+M,ppos+2*M-1)).reshaped(N,L);
    //std::cout << mat << "\n"; 
    // translate & populate the kernel
    for (int T=0; T<num_tsymms_; ++T) {
      kernel_.block(block_row+row,0,N,L) = row_translate(mat, T);
      row += N;
    }
    // next block
    block_row += num_visible_units_;
    ppos += 2*M;
  }

  // The right half block is same as the left half, assuming spin-flip symmetry
  kernel_.block(0,L,num_hidden_units_,L) = kernel_.block(0,0,num_hidden_units_,L);

  // for checking
  /*std::ios state(NULL);
  state.copyfmt(std::cout);
  std::cout<<std::fixed<<std::setprecision(2)<<std::right;
  std::cout << "\n'kernel' parameters:\n";
  for (int i=0; i<num_hidden_units_; ++i) {
    for (int j=0; j<num_visible_units_; ++j) {
      std::cout << kernel_(i,j) << "  ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
  std::cout.copyfmt(state);
  */

  return 0;
}

int RBM::update_hbias_params(const RealVector& params)
{
  assert(params.size()==num_hbias_params_);

  int N = num_basis_sites_;
  Vector vec(N);
  int block_row = 0;
  int ppos = 0;
  for (int block=0; block<num_hblocks_; ++block) {
    int row = 0;
    // Upper (spin-up) half
    vec = params(Eigen::seqN(ppos,N));
    for (int i=0; i<num_tsymms_; ++i) {
      for (int n=0; n<num_basis_sites_; ++n) {
        hbias_[block_row+row] = vec[n];
        row++;
      }
    }
    // Lower (spin-dn) half
    vec = params(Eigen::seqN(ppos+N,N));
    for (int i=0; i<num_tsymms_; ++i) {
      for (int n=0; n<num_basis_sites_; ++n) {
        hbias_[block_row+row] = vec[n];
        row++;
      }
    }
    // next block
    block_row += num_visible_units_;
    ppos += 2*N;
  }

  // for checking
  /*
  std::ios state(NULL);
  state.copyfmt(std::cout);
  std::cout<<std::fixed<<std::setprecision(2)<<std::right;
  std::cout << "\n'hbias' parameters:\n";
  for (int i=0; i<num_hidden_units_; ++i) {
    std::cout << hbias_[i] << "\n";
  }
  std::cout << "\n";
  std::cout.copyfmt(state);
  */

  return 0;
}

Matrix RBM::row_translate(const RealMatrix& mat, const int& T) const
{
  assert(mat.cols() <= num_sites_);
  Matrix tmat(mat.rows(), mat.cols());
  for (int i=0; i<mat.rows(); ++i) {
    for (int j=0; j<mat.cols(); ++j) {
      int tj = tsymm_map_(T,j);
      tmat(i,tj) = mat(i,j);
    }
  }
  return tmat;
} 

int RBM::add_layer(const int& units, const std::string& activation, 
  const int& input_dim)
{
  throw std::invalid_argument("RBM::add_layer: unidefined function");
}

int RBM::add_sign_layer(const int& input_dim)
{
  throw std::invalid_argument("RBM::add_sign_layer: unidefined function");
}

void RBM::init_parameters(random_engine& rng, const double& sigma) 
{
  std::normal_distribution<double> random_normal(0.0,sigma);
  for (int i=0; i<num_params_; ++i) {
    pvector_[i] = random_normal(rng);
  }
  // kernel parameters
  update_kernel_params(pvector_(Eigen::seqN(0,num_kernel_params_)));
  // bias
  update_hbias_params(pvector_.tail(num_hbias_params_));
}

void RBM::init_parameter_file(const std::string& prefix)
{
  prefix_ = prefix;
  boost::filesystem::path prefix_dir(prefix_);
  boost::filesystem::create_directories(prefix_dir);
}

void RBM::save_parameters(void) const
{
  std::string fname = prefix_+"/rbm_L"+std::to_string(num_sites_)+".txt";
  std::cout << "RBM:: saving parameters to file: '"<<fname<<"'\n";
  std::ofstream fs(fname);
  if (fs.is_open()) {
    fs << "#  RBM: visible_units = "<<num_visible_units_<<" , hidden_dim = "<<num_hidden_units_ <<"\n";
    fs << "#  Hbias        |   Kernel\n";
    fs << std::scientific << std::uppercase << std::setprecision(8) << std::right;
    for (int i=0; i<num_hidden_units_; ++i) {
      fs << std::setw(16)<< hbias_[i] << " ";
      for (int j=0; j<num_visible_units_; ++j) {
        fs << std::setw(16) << kernel_(i,j);
      }
      fs << "\n";
    }
    fs.close();
  }
  else {
    throw std::range_error("RBM::save_parameters: file open failed");
  }
}

void RBM::load_parameters(const std::string& load_path)
{
  boost::char_separator<char> space(" ");
  boost::tokenizer<boost::char_separator<char> >::iterator it;
  std::string line;
  std::string::size_type pos;
  std::string fname = prefix_+"/rbm_L"+std::to_string(num_sites_)+".txt";
  std::cout << "RBM:: loading parameters from file: '"<<fname<<"'\n";
  std::ifstream fin(fname);
  if (fin.is_open()) {
    int row = 0;
    while (std::getline(fin,line)) {
      // skip comments & blank lines
      pos = line.find_first_of("#");
      if (pos != std::string::npos) line.erase(pos);
      if (line.find_first_not_of(" ") == std::string::npos) continue;
      boost::tokenizer<boost::char_separator<char> > tokens(line, space);
      if (std::distance(tokens.begin(), tokens.end()) != (1+num_visible_units_)) {
        throw std::range_error("RBM::load_parameters: incorrect number of columns\n");
      }
      it=tokens.begin();
      // load 
      hbias_[row] = std::stod(*it);
      int col = 0;
      for (++it; it!=tokens.end(); ++it) {
        kernel_(row,col++) = std::stod(*it);
      }
      row++;
    }
    fin.close();
    // update parameter vector 
    int i, j;
    for (int n=0; n<num_kernel_params_; ++n) {
      std::tie(i,j) = kernel_params_map_[n][0];
      pvector_[n] = kernel_(i,j);
      std::cout << pvector_[n] << "\n";
    }
    std::cout << "\n";
    for (int n=0; n<num_hbias_params_; ++n) {
      i = bias_params_map_[n][0];
      pvector_[num_kernel_params_+n] = hbias_[i];
      std::cout << pvector_[num_kernel_params_+n] << "\n";
    }
    std::cout << "\n";
  }
  else {
    throw std::range_error("RBM::load_parameters: file open failed");
  }
}

const double& RBM::get_parameter(const int& id) const
{
  if (id < num_params_) {
    return pvector_[id];
  }
  throw std::out_of_range("RBM::get_parameter: out-of-range 'id'");
}

void RBM::get_parameters(Vector& pvec) const
{
  pvec = pvector_;
}

void RBM::get_parameter_names(std::vector<std::string>& pnames, const int& pos) const
{
  for (int i=1; i<num_layers_; ++i) {
    int start_pos = pos+num_params_fwd_[i-1];
    layers_[i]->get_parameter_names(pnames,start_pos);
  }
}

void RBM::get_parameter_values(RealVector& pvalues, const int& pos) const
{
  for (int i=1; i<num_params_; ++i) {
    pvalues[pos+i] = pvector_[i];
  }
}

void RBM::get_parameter_vector(std::vector<double>& pvalues, const int& pos) const
{
  for (int i=1; i<num_params_; ++i) {
    pvalues[pos+i] = pvector_[i];
  }
}

void RBM::update_parameters(const RealVector& pvec, const int& start_pos)
{
  for (int i=1; i<num_params_; ++i) {
    pvector_[i] = pvec[start_pos+i];
  }
  // kernel parameters
  update_kernel_params(pvector_(Eigen::seqN(0,num_kernel_params_)));
  // bias
  update_hbias_params(pvector_.tail(num_hbias_params_));
}

void RBM::update_parameter(const int& id, const double& value)
{
  if (id < num_kernel_params_) {
    for (const auto& idx : kernel_params_map_[id]) {
      kernel_(idx.first, idx.second) = value;
    }
  }
  throw std::out_of_range("RBM::update_parameter: out-of-range 'id'");
}

void RBM::do_update_run(const Vector& input)
{
  input_ = input;
  lin_output_ = kernel_*input_ + hbias_;
  output_[0] = 1.0;
  for (int i=0; i<num_hidden_units_; i++) {
    cosh_output_[i] = std::cosh(lin_output_[i]);
    output_[0] *= cosh_output_[i];
  }
}

void RBM::do_update_run(const Vector& new_input, const std::vector<int> new_elems) 
{
  // changes to lin outputs
  if (new_elems.size()==0) return;
  for (int i=0; i<num_hidden_units_; i++) {
    for (const auto& j : new_elems) {
      lin_output_[i] += kernel_(i,j)*(new_input[j] - input_[j]);
    }
  }
  // output
  output_[0] = 1.0;
  for (int i=0; i<num_hidden_units_; i++) {
    cosh_output_[i] = std::cosh(lin_output_[i]);
    output_[0] *= cosh_output_[i];
  }
  // update inputs
  for (const auto& j : new_elems) {
    input_[j] = new_input[j];
  }
}

Vector RBM::get_new_output(const Vector& input) const
{
  /* Does NOT change the state of the network */
  RealVector xout = kernel_*input + hbias_;
  double val = 1.0;
  for (int i=0; xout.size(); ++i) {
    val *= std::cosh(xout[i]);
  }
  Vector new_output(1);
  new_output(1) = val;
  return new_output;
}

Vector RBM::get_new_output(const Vector& new_input, const std::vector<int> new_elems) const
{
  RealVector xout = lin_output_;
  for (int i=0; i<num_hidden_units_; i++) {
    for (const auto& j : new_elems) {
      xout[i] += kernel_(i,j)*(new_input[j] - input_[j]);
    }
  }
  double val = 1.0;
  for (int i=0; xout.size(); ++i) {
    val *= std::cosh(xout[i]);
  }
  Vector new_output(1);
  new_output(1) = val;
  return new_output;
}

const Matrix& RBM::get_gradient(void) const
{
  // tanh outputs
  RealVector tanh_output_(num_hidden_units_);
  for (int i=0; i<num_hidden_units_; ++i) {
    double x = cosh_output_[i];
    tanh_output_[i] = std::sqrt(1.0-1.0/(x*x));
  }

  // Gradient wrt to kernel parameters
  int i, j;
  RealVector d_output_(num_hidden_units_);
  for (int n=0; n<num_kernel_params_; ++n) {
    d_output_.setZero();
    for (const auto& elem: kernel_params_map_[n]) {
      std::tie(i,j) = elem;
      d_output_[i] += input_[j];
    }
    // gradient
    double sum = 0.0;
    for (int i=0; i<num_hidden_units_; ++i) {
      if (std::abs(d_output_[i])>1.0E-12) {
        sum += tanh_output_[i] * d_output_[i];
      }
    }
    gradient_(n,0) = output_[0]*sum;
  }

  // Gradient wrt to bias parameters
  for (int n=0; n<num_hbias_params_; ++n) {
    double sum = 0.0;
    for (const int& i: bias_params_map_[n]) {
      sum += tanh_output_[i];
    }
    gradient_(num_kernel_params_+n,0) = output_[0]*sum;
  }

  return gradient_;
}


} // end namespace ann

