/*---------------------------------------------------------------------------
* Copyright (C) 2015-2016 by Amal Medhi <amedhi@iisertvm.ac.in>.
* All rights reserved.
* Author: Amal Medhi
* Date:   2016-03-11 13:02:35
* Last Modified by:   Amal Medhi, amedhi@macbook
* Last Modified time: 2017-04-12 21:16:21
*----------------------------------------------------------------------------*/
#include <cmath>
#include "model.h"
#include <boost/algorithm/string.hpp>

namespace model {

int Hamiltonian::define_model(const input::Parameters& inputs, 
  const lattice::Lattice& lattice)
{
  // define the models 
  double defval;
  std::string name; //, matrixelem, op, qn, site, src, tgt, fact;
  CouplingConstant cc;
  int info;
  model_name = inputs.set_value("model", "HUBBARD");
  boost::to_upper(model_name);

  if (model_name == "HUBBARD") {
    mid = model_id::HUBBARD;
    // model parameters
    add_parameter(name="t", defval=1.0, inputs);
    add_parameter(name="U", defval=0.0, inputs);
    // bond operator terms
    add_bondterm(name="hopping", cc="-t", op::spin_hop());
    add_siteterm(name="hubbard", cc="U", op::hubbard_int());
  }

  else if (model_name == "TJ") {
    mid = model_id::TJ;
    int nowarn;
    if (inputs.set_value("no_double_occupancy",true,nowarn))
      set_no_dbloccupancy();
    // model parameters
    add_parameter(name="t", defval=1.0, inputs);
    add_parameter(name="J", defval=0.0, inputs);
    // bond operator terms
    add_bondterm(name="hopping", cc="-t", op::spin_hop());
    add_bondterm(name="exchange", cc="J", op::sisj_plus());
  }

  //------------------------HUBBARD IONIC-------------------------------------
  else if (model_name == "HUBBARD_IONIC") {
    mid = model_id::HUBBARD_IONIC;
    if ( (lattice.id()==lattice::lattice_id::SQUARE_2SITE) ||
         (lattice.id()==lattice::lattice_id::SQUARE_4SITE)) {

      add_parameter(name="t", defval=1.0, inputs);
      add_parameter(name="tp", defval=1.0, inputs);
      add_parameter(name="W", defval=0.0, inputs);
      add_parameter(name="U", defval=0.0, inputs);

      // A-B hopping term
      cc.create(6);
      cc.add_type(0,"-t");
      cc.add_type(1,"-t");
      cc.add_type(2,"0");
      cc.add_type(3,"0");
      cc.add_type(4,"0");
      cc.add_type(5,"0");
      add_bondterm(name="hop-AB", cc, op::spin_hop());

      // A-A hopping term
      cc.create(6);
      cc.add_type(0,"0");
      cc.add_type(1,"0");
      cc.add_type(2,"-tp");
      cc.add_type(3,"-tp");
      cc.add_type(4,"0");
      cc.add_type(5,"0");
      add_bondterm(name="hop-AA", cc, op::spin_hop());

      // B-B hopping term
      cc.create(6);
      cc.add_type(0,"0");
      cc.add_type(1,"0");
      cc.add_type(2,"0");
      cc.add_type(3,"0");
      cc.add_type(4,"-tp");
      cc.add_type(5,"-tp");
      add_bondterm(name="hop-BB", cc, op::spin_hop());

      // ionic potential
      cc.create(2);
      cc.add_type(0, "-0.5*W");
      cc.add_type(1, "0.5*W");
      add_siteterm(name="ni_sigma", cc, op::ni_sigma());

      // Hubbard interaction
      add_siteterm(name="hubbard", cc="U", op::hubbard_int());
    }

    else {
      throw std::range_error("*error: modellibrary: model not defined for this lattice");
    }
  }

  else if (model_name == "TJ_IONIC") {
    mid = model_id::TJ_IONIC;
    if ((lattice.id()==lattice::lattice_id::SQUARE_2SITE) ||
        (lattice.id()==lattice::lattice_id::SQUARE_4SITE)) {
      add_parameter(name="t", defval=1.0, inputs);
      add_parameter(name="tp", defval=1.0, inputs);
      add_parameter(name="W", defval=0.0, inputs);
      add_parameter(name="U", defval=0.0, inputs);

      // projection operator
      if (inputs.set_value("projection",true,info)) {
        ProjectionOp pjn;
        pjn.set({0,projection_t::HOLON}, {1,projection_t::DOUBLON});
        set_projection_op(pjn);
      }

      /*
      // hopping term - split into NN & NNN
      cc = CouplingConstant({0,"-t"},{1,"-t"},{2,"0"});
      add_bondterm(name="hopping-1", cc, op::spin_hop());
      cc = CouplingConstant({0,"0"},{1,"0"},{2,"-tp"});
      add_bondterm(name="hopping-2", cc, op::spin_hop());
      */

      // A-B hopping term
      cc.create(6);
      cc.add_type(0,"-t");
      cc.add_type(1,"-t");
      cc.add_type(2,"0");
      cc.add_type(3,"0");
      cc.add_type(4,"0");
      cc.add_type(5,"0");
      add_bondterm(name="hop-AB", cc, op::spin_hop());

      // A-A hopping term
      cc.create(6);
      cc.add_type(0,"0");
      cc.add_type(1,"0");
      cc.add_type(2,"-tp");
      cc.add_type(3,"-tp");
      cc.add_type(4,"0");
      cc.add_type(5,"0");
      add_bondterm(name="hop-AA", cc, op::spin_hop());

      // B-B hopping term
      cc.create(6);
      cc.add_type(0,"0");
      cc.add_type(1,"0");
      cc.add_type(2,"0");
      cc.add_type(3,"0");
      cc.add_type(4,"-tp");
      cc.add_type(5,"-tp");
      add_bondterm(name="hop-BB", cc, op::spin_hop());

      // Exchange term
      cc.create(6);
      cc.add_type(0, "2.0*t*t/(U+W)");
      cc.add_type(1, "2.0*t*t/(U+W)");
      cc.add_type(2, "4.0*tp*tp/U");
      cc.add_type(3, "4.0*tp*tp/U");
      cc.add_type(4, "4.0*tp*tp/U");
      cc.add_type(5, "4.0*tp*tp/U");
      add_bondterm(name="exchange", cc, op::sisj_plus());

      // NN density-density terms
      cc.create(6);
      cc.add_type(0, "t*t/(U+W)-2.0*t*t/W");
      cc.add_type(1, "t*t/(U+W)-2.0*t*t/W");
      cc.add_type(2, "0");
      cc.add_type(3, "0");
      cc.add_type(4, "0");
      cc.add_type(5, "0");
      add_bondterm(name="ni_nj", cc, op::ni_nj());

      // Hubbard interaction
      cc.create(2);
      cc.add_type(0, "0.5*(U-W)");
      cc.add_type(1, "0.5*(U-W)");
      add_siteterm(name="hubbard", cc, op::hubbard_int());

      // extra onsite terms
      cc.create(2);
      cc.add_type(0, "8.0*tp*tp/U+2.0*t*t/W");
      cc.add_type(1, "6.0*t*t/W-0.5*(U-W)-2.0*t*t/(U+W)");
      add_siteterm(name="ni_sigma", cc, op::ni_sigma());

      // extra terms
      //add_siteterm(name="hubbard", cc="U", op::hubbard_int());
    }
    else {
      throw std::range_error("*error: modellibrary: model not defined for this lattice");
    }
  }

  /*------------- undefined lattice--------------*/
  else {
    throw std::range_error("*error: modellibrary: undefined model");
  }

  // if the model has site disorder
  /*
  if (site_disorder) {
    add_disorder_term(name="disorder", op::ni_sigma());
  }*/
  
  return 0;
}

int Hamiltonian::construct(const input::Parameters& inputs, 
  const lattice::Lattice& lattice)
{
  init(lattice);
  define_model(inputs, lattice);
  finalize(lattice);
  return 0;
}


} // end namespace model
