/*---------------------------------------------------------------------------
* @Author: Amal Medhi, amedhi@mbpro
* @Date:   2019-09-26 13:53:41
* @Last Modified by:   Amal Medhi
* @Last Modified time: 2022-06-22 16:03:58
* Copyright (C) Amal Medhi, amedhi@iisertvm.ac.in
*----------------------------------------------------------------------------*/
#include "./particle.h"

namespace vmc {

//-------------------Site Occupancy--------------------------------
void SiteOccupancy::setup(const lattice::LatticeGraph& graph, 
	const SysConfig& config)
{
  MC_Observable::switch_on();
  if (setup_done_) return;
  num_sites_ = graph.num_sites();
  num_basis_sites_ = graph.lattice().num_basis_sites();
  num_particles_ = config.num_particles();
  std::vector<std::string> elem_names(num_basis_sites_);
  for (int i=0; i<num_basis_sites_; ++i) {
  	std::ostringstream ss;
  	ss << "site-"<<i;
  	elem_names[i] = ss.str();
  }
  this->resize(elem_names.size(), elem_names);
  //this->set_have_total();
  config_value_.resize(elem_names.size());
  setup_done_ = true;
}

void SiteOccupancy::measure(const lattice::LatticeGraph& graph, 
	const SysConfig& config) 
{
  IntVector matrix_elem(num_basis_sites_);
  IntVector num_subsites(num_basis_sites_);
  matrix_elem.setZero();
  num_subsites.setZero();
  for (auto s=graph.sites_begin(); s!=graph.sites_end(); ++s) {
    int site = graph.site(s);
    int basis = graph.site_uid(s);
    matrix_elem(basis) += config.apply(model::op::ni_sigma(),site);
    num_subsites(basis) += 1;
  }
  for (int i=0; i<num_basis_sites_; ++i) {
  	//config_value_[i] = static_cast<double>(matrix_elem[i])/num_particles_;
    config_value_[i] = static_cast<double>(matrix_elem[i])/num_subsites[i];
  }
  // add to databin
  *this << config_value_;
}

//-------------------Momentum occupancy (n[k])--------------------------------
void MomentumOccupancy::setup(const lattice::LatticeGraph& graph, 
  const SysConfig& config)
{
  MC_Observable::switch_on();
  if (setup_done_) return;
  num_sites_ = graph.num_sites();
  num_basis_sites_ = graph.lattice().num_basis_sites();
  num_kpoints_ = config.wavefunc().blochbasis().num_kpoints();
  exp_ikr_.resize(num_kpoints_,num_sites_);
  kvals_.resize(num_kpoints_);
  nk_.resize(num_kpoints_);
  for (auto& elem : nk_) elem.resize(num_basis_sites_,num_basis_sites_);

  for (int k=0; k<num_kpoints_; ++k) {
    Vector3d kvec = config.wavefunc().blochbasis().kvector(k);
    kvals_[k] = kvec;
    for (auto s=graph.sites_begin(); s!=graph.sites_end(); ++s) {
      int i = graph.site(s);
      Vector3d R = graph.site_cellcord(s);
      exp_ikr_(k,i) = std::exp(ii()*kvec.dot(R)); 
    }
  }
  std::vector<std::string> elem_names;
  for (int i=0; i<num_basis_sites_; ++i) {
    for (int j=0; j<num_basis_sites_; ++j) {
      std::ostringstream ss;
      ss << "nk-"<<i<<j;
      elem_names.push_back(ss.str());
    }
  }
  this->resize(num_sites_,elem_names);
  //this->set_have_total();
  config_value_.resize(num_sites_);
  setup_done_ = true;
}


void MomentumOccupancy::measure(const lattice::LatticeGraph& graph, 
  const SysConfig& config) 
{
  // \sum_s <c^\dag_{is} c_{js}>
  ComplexMatrix N(num_sites_,num_sites_);
  for (auto i=0; i<num_sites_; ++i) {
    for (auto j=0; j<num_sites_; ++j) {
      N(i,j) = config.apply(model::op::cdagc_sigma(),i,j,1,1.0);
    }
  }
  // nk values
  RealMatrix nk(num_basis_sites_,num_basis_sites_);
  double norm = 1.0/(2*num_kpoints_);
  int n = 0;
  for (int k=0; k<num_kpoints_; ++k) {
    nk.setZero();
    for (auto s1=graph.sites_begin(); s1 != graph.sites_end(); ++s1) {
      int i = graph.site(s1);
      int a = graph.site_uid(s1);
      for (auto s2=graph.sites_begin(); s2 != graph.sites_end(); ++s2) {
        int j = graph.site(s2);
        int b = graph.site_uid(s2);
        nk(a,b) += std::real(exp_ikr_(k,i)*std::conj(exp_ikr_(k,j))*N(i,j));
      }
    }
    // linearize 
    for (auto a=0; a<num_basis_sites_; ++a) {
      for (auto b=0; b<num_basis_sites_; ++b) {
        config_value_[n++] = nk(a,b)*norm;
      }
    }
  }
  //std::cout << "sum = " <<  config_value_.sum() << "\n";
  //getchar();

  /*
  for (auto& elem : nk_) elem.setZero();
  for (auto s1=graph.sites_begin(); s1 != graph.sites_end(); ++s1) {
    int i = graph.site(s1);
    int ia = graph.site_uid(s1);
    for (auto s2=graph.sites_begin(); s2 != graph.sites_end(); ++s2) {
      int j = graph.site(s2);
      int ja = graph.site_uid(s2);
      amplitude_t ampl = config.apply(model::op::cdagc_sigma(),i,j,1,1.0);
      for (int k=0; k<num_kpoints_; ++k) {
        nk_[k](ia,ja) += std::real(exp_ikr_(k,i)*std::conj(exp_ikr_(k,j))*ampl);
      }
    }
  }
  double norm = 1.0/(2*num_kpoints_);
  int i = 0;
  double sum = 0.0;
  for (int k=0; k<num_kpoints_; ++k) {
    for (int ia=0; ia<num_basis_sites_; ++ia) {
      for (int ja=0; ja<num_basis_sites_; ++ja) {
        config_value_[i] = nk_[k](ia,ja)*norm;
        sum += config_value_[i];
        //std::cout << "nk["<<k<<"] = "<<config_value_[i] << "\n";
        i++;
      }
    }
  }
  std::cout << "sum = " <<  sum << "\n";
  */

  // add to databin
  *this << config_value_;
}

void MomentumOccupancy::print_heading(const std::string& header,
  const std::vector<std::string>& xvars) 
{
  if (!is_on()) return;
  if (heading_printed_) return;
  if (!replace_mode_) return;
  if (!is_open()) open_file();
  fs_ << header;
  fs_ << "# Results: " << name() << "\n";
  fs_ << "#" << std::string(72, '-') << "\n";

  fs_ << "# ";
  fs_ << std::left;
  for (const auto& p : xvars) fs_ << std::setw(14)<<p.substr(0,14);
  fs_ << std::endl;

  fs_ << "# ";
  fs_ << std::setw(6)<<"ik";
  for (const auto& name : elem_names_) 
    fs_ << std::setw(14)<<name<<std::setw(11)<<"err";
  fs_ << std::setw(9)<<"samples"<<std::setw(12)<<"converged"<<std::setw(6)<<"tau";
  fs_ << std::endl;
  fs_ << "#" << std::string(72, '-') << "\n";

  heading_printed_ = true;
  close_file();
}

void MomentumOccupancy::print_result(const std::vector<double>& xvals) 
{
  if (!is_on()) return;
  if (!is_open()) open_file();
  fs_ << std::right;
  fs_ << std::scientific << std::uppercase << std::setprecision(6);

  fs_ << "# ";
  for (const auto& p : xvals) fs_ << std::setw(14) << p;
  fs_ << std::endl;

  // nk in correct format
  int n = 0;
  for (int k=0; k<num_kpoints_; ++k) {
    fs_ << std::left;
    fs_ << std::setw(6) << k; 
    fs_ << std::right;
    fs_ << std::setw(14) << kvals_[k](0);
    fs_ << std::setw(14) << kvals_[k](1);
    fs_ << std::setw(14) << kvals_[k](2);
    for (int a=0; a<num_basis_sites_; ++a) {
      for (int b=0; b<num_basis_sites_; ++b) {
        fs_ << MC_Data::result_str(n++);
        fs_ << std::endl;
      }
    }
  }
  fs_ << std::endl;
  fs_ << std::flush;
  close_file();
}


} // end namespave vmc








