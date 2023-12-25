/*---------------------------------------------------------------------------
* Author: Amal Medhi
* @Author: Amal Medhi
* @Date:   2023-08-26 13:41:42
* @Last Modified by:   Amal Medhi
* @Last Modified time: 2023-09-11 23:18:43
*----------------------------------------------------------------------------*/
#include "./bandstruct.h"

namespace vmc {

void BandStruct::setup(const lattice::Lattice& lattice, const SysConfig& config)
{
  MC_Observable::switch_on();
  if (setup_done_) return;

  // setup k-points along symmetry path
  num_bands_ = config.wavefunc().blochbasis().subspace_dimension();
  symm_kpoints_.clear();
  symm_pidx_.clear();
  symm_pname_.clear();
  Vector3d vec_b1 = config.wavefunc().blochbasis().vector_b1();
  Vector3d vec_b2 = config.wavefunc().blochbasis().vector_b2();
  Vector3d vec_b3 = config.wavefunc().blochbasis().vector_b3();
  //std::cout << "b1 = " << vec_b1.transpose() << "\n";
  //std::cout << "b2 = " << vec_b2.transpose() << "\n";
  //std::cout << "b3 = " << vec_b3.transpose() << "\n";

  if (lattice.id()==lattice::lattice_id::SQUARE ||
      lattice.id()==lattice::lattice_id::SQUARE_2SITE ||
      lattice.id()==lattice::lattice_id::SQUARE_4SITE
     ) {
    Vector3d Gamma = Vector3d(0,0,0);
    Vector3d X = 0.5*vec_b1;
    Vector3d M = X+0.5*vec_b2;
    //std::cout << "X=" << X << "\n";
    //std::cout << "M=" << M << "\n";
    int N = 100;
    int i = 0;
    int idx = 0;
    //---------------------------------
    symm_pidx_.push_back(idx);
    symm_pname_.push_back("Gamma");
    Vector3d step = (X-Gamma)/N;
    for (i=0; i<N; ++i) symm_kpoints_.push_back(Gamma+i*step);
    //---------------------------------
    idx += i;
    symm_pidx_.push_back(idx);
    symm_pname_.push_back("X");
    step = (M-X)/N;
    for (i=0; i<N; ++i) symm_kpoints_.push_back(X+i*step);
    //---------------------------------
    idx += i;
    symm_pidx_.push_back(idx);
    symm_pname_.push_back("M");
    step = (Gamma-M)/N;
    for (i=0; i<N; ++i) symm_kpoints_.push_back(M+i*step);
    //---------------------------------
    idx += i;
    symm_pidx_.push_back(idx);
    symm_pname_.push_back("Gamma");
  }

  else {
    throw std::range_error("BandStruct::setup: not implemented for this lattice");
  }

  Ekn_.resize(symm_kpoints_.size(),num_bands_);
  this->resize(1); // not used, actually
  setup_done_ = true;
  computation_done_ = false;
}

void BandStruct::reset(void) 
{
  computation_done_ = false;
}

void BandStruct::measure(const lattice::Lattice& lattice, const SysConfig& config)
{
  if (computation_done_) return;
  // compute dispersion
  Eigen::SelfAdjointEigenSolver<ComplexMatrix> es;
  for (int k=0; k<symm_kpoints_.size(); ++k) {
    Vector3d kvec = symm_kpoints_[k];
    config.wavefunc().mf_model().construct_kspace_block(kvec);
    es.compute(config.wavefunc().mf_model().quadratic_spinup_block(), Eigen::EigenvaluesOnly);
    Ekn_.row(k) = es.eigenvalues().transpose();
  }
  computation_done_ = true;
}

void BandStruct::print_heading(const std::string& header,
  const std::vector<std::string>& xvars) 
{
  if (!is_on()) return;
  if (heading_printed_) return;
  if (!replace_mode_) return;
  if (!is_open()) open_file();
  xvars_ = xvars;
  fs_ << header;
  fs_ << "# Results: " << name() << "\n";
  fs_ << "#" << std::string(72, '-') << "\n";
  fs_ << "# Special k-points:\n";
  fs_ << "# ";
  fs_ << std::right;
  for (const auto& name : symm_pname_) fs_<<std::setw(8)<<name; 
  fs_ << std::endl;
  fs_ << "# ";
  for (const auto& idx : symm_pidx_) fs_<<std::setw(8)<<idx; 
  fs_ << std::endl;
  fs_ << "#" << std::string(72, '-') << "\n";

  fs_ << std::flush;
  heading_printed_ = true;
  close_file();
}

void BandStruct::print_result(const std::vector<double>& xvals) 
{
  if (!is_on()) return;
  if (!is_open()) open_file();
  fs_ << std::right;
  fs_ << std::scientific << std::uppercase << std::setprecision(6);

  for (int i=0; i<xvars_.size(); ++i) {
    fs_ << "# ";
    fs_ << xvars_[i].substr(0,14)<<" =";
    fs_ << std::setw(14)<<xvals[i];
    fs_ << std::endl;
  }

  fs_ << "#" << std::string(72, '-') << "\n";
  fs_ << "# ";
  fs_ << std::left;
  //for (const auto& p : xvars) fs_ << std::setw(14)<<p.substr(0,14);
  fs_ << std::setw(6)<<"k"<<std::setw(14)<<"kx"<<std::setw(14)<<"ky"<<std::setw(14)<<"kz";
  for (int n=0; n<num_bands_; ++n) {
    fs_<<std::setw(14)<<"Ek("+std::to_string(n)+")"; 
  }
  fs_ << std::endl;
  fs_ << "#" << std::string(72, '-') << "\n";

  fs_ << std::right;
  for (int k=0; k<symm_kpoints_.size(); ++k) {
    Vector3d kvec = symm_kpoints_[k];
    fs_<<std::setw(6) << k; 
    fs_<<std::setw(14)<<kvec(0)<<std::setw(14)<<kvec(1)<<std::setw(14)<<kvec(2); 
    for (int n=0; n<num_bands_; ++n) {
      fs_<<std::setw(14)<<Ekn_(k,n); 
    }
    fs_ << std::endl; 
  }
  fs_ << std::endl; 
  fs_ << std::flush;
  close_file();
}


} // end namespace vmc