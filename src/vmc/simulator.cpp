/*---------------------------------------------------------------------------
* Copyright (C) 2015-2016 by Amal Medhi <amedhi@iisertvm.ac.in>.
* All rights reserved.
* Author: Amal Medhi
* Date:   2016-03-09 15:27:50
* Last Modified by:   Amal Medhi, amedhi@macbook
* Last Modified time: 2017-05-20 11:14:38
*----------------------------------------------------------------------------*/
#include <iomanip>
#include "simulator.h"

namespace vmc {

Simulator::Simulator(const input::Parameters& inputs) : vmc(inputs)
{
  optimization_mode_ = inputs.set_value("optimizing_run",false);
  if (optimization_mode_) {
    sreconf.init(inputs, vmc);
    //vmc.set_box_constraints();
    //nlopt_.init(inputs, vmc);
  }
}

int Simulator::run(const input::Parameters& inputs) 
{
  // optimization run
  if (optimization_mode_) {
    if (!inputs.have_option_quiet()) std::cout << " initing optimizing run\n";
    //vmc.init(inputs, run_mode::energy_function, false);
    //nlopt_.optimize(vmc);
    vmc.init(inputs, run_mode::sr_function, true);
    if (sreconf.optimize(vmc)) {
      vmc.run_simulation(sreconf.optimal_parms());
      vmc.print_results();
    }
    return 0;
  }

  // normal run
  if (!inputs.have_option_quiet()) std::cout << " initing vmc run\n";
  vmc.init(inputs, run_mode::normal);
  vmc.run_simulation();
  vmc.print_results();
  return 0;
}

// parallel run
int Simulator::run(const input::Parameters& inputs, 
  const scheduler::mpi_communicator& mpi_comm)
{
  std::vector<int> pending_msg(mpi_comm.slave_max_id()+1, 0);
  int pending_messages = 0;
  if (mpi_comm.is_master()) {
    vmc.init(inputs,run_mode::normal);
    vmc.do_warmup();
    //std::cout << "rank = " << mpi_comm.rank() << "\n";
    for (const auto& p : mpi_comm.slave_procs()) {
      mpi_comm.isend(p, MP_init_simulation);
    }
    while(vmc.not_done()) {
      // do own work
      vmc.do_steps(10);
      // slaves, send your data
      for (const auto& p : mpi_comm.slave_procs()) {
        if (!pending_msg[p]) {
          mpi_comm.isend(p, MP_poll_results);
          pending_msg[p] = 1;
          pending_messages++;
        }
      }
      // check message
      while(pending_messages) {
        if (mpi::mpi_status_opt msg = mpi_comm.iprobe()) {
          mpi_comm.recv(msg->source(),msg->tag());
          pending_msg[msg->source()] = 0;
          pending_messages--;
          //no_pending_msg[msg->source()] = true;
          switch (msg->tag()) {
            case MP_run_results: 
              std::cout << "master: recv MP_run_results from "<<msg->source()<<"\n";
              break;
            default: break;
          }
        }
        else break;
      }
    }
    for (const auto& p : mpi_comm.slave_procs()) {
      if (pending_msg[p]) {
        mpi::mpi_status msg = mpi_comm.probe(p);
        mpi_comm.recv(msg.source(),msg.tag());
        pending_messages--;
      }
      mpi_comm.isend(p, MP_quit_simulation);
    }
    return 0;
  }
  /* slaves */
  else {
    bool not_done = true;
    while(true) {
      auto msg = mpi_comm.probe();
      mpi_comm.recv(msg.source(),msg.tag());
      switch (msg.tag()) {
        case MP_init_simulation:
          vmc.init(inputs,run_mode::normal,true);
          vmc.do_warmup();
          break;
        case MP_quit_simulation:
          return 0;
        default: continue;
      }
      // vmc steps
      while(not_done) {
        if (mpi::mpi_status_opt next_msg = mpi_comm.iprobe()) {
          mpi_comm.recv(next_msg->source(),next_msg->tag());
          switch(next_msg->tag()) {
            case MP_poll_results:
              mpi_comm.isend(next_msg->source(),MP_run_results);
              std::cout << "slave-"<<mpi_comm.rank()<<": recv MP_poll_results\n";
              not_done = true;
              break;
            case MP_halt_simulation:
              not_done = false;
              break;
            case MP_quit_simulation:
              std::cout << "slave-"<<mpi_comm.rank()<<": recv MP_quit_simulation\n";
              return 0;
            default: continue;
          }
        }
        else {
          vmc.do_steps(10);
        }
      }
    }
  }



  return 0;
}

void Simulator::print_copyright(std::ostream& os)
{
  VMC::copyright_msg(os);
}


} // end namespace mc
