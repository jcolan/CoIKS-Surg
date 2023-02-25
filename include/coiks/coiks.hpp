/******************************************************************************
# coiks.hpp:  Concurrent Inverse Kinematics solver                            #
# Copyright (c) 2023                                                          #
# Hasegawa Laboratory at Nagoya University                                    #
#                                                                             #
# Redistribution and use in source and binary forms, with or without          #
# modification, are permitted provided that the following conditions are met: #
#                                                                             #
#     - Redistributions of source code must retain the above copyright        #
#       notice, this list of conditions and the following disclaimer.         #
#     - Redistributions in binary form must reproduce the above copyright     #
#       notice, this list of conditions and the following disclaimer in the   #
#       documentation and/or other materials provided with the distribution.  #
#     - Neither the name of the Hasegawa Laboratory nor the                   #
#       names of its contributors may be used to endorse or promote products  #
#       derived from this software without specific prior written permission. #
#                                                                             #
# This program is free software: you can redistribute it and/or modify        #
# it under the terms of the GNU Lesser General Public License LGPL as         #
# published by the Free Software Foundation, either version 3 of the          #
# License, or (at your option) any later version.                             #
#                                                                             #
# This program is distributed in the hope that it will be useful,             #
# but WITHOUT ANY WARRANTY; without even the implied warranty of              #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the                #
# GNU Lesser General Public License LGPL for more details.                    #
#                                                                             #
# You should have received a copy of the GNU Lesser General Public            #
# License LGPL along with this program.                                       #
# If not, see <http://www.gnu.org/licenses/>.                                 #
#                                                                             #
# #############################################################################
#                                                                             #
#   Author: Jacinto Colan, email: colan@robo.mein.nagoya-u.ac.jp              #
#                                                                             #
# ###########################################################################*/

#ifndef COIKS_HPP
#define COIKS_HPP

#define PINOCCHIO_URDFDOM_TYPEDEF_SHARED_PTR

#include <cmath>
#include <memory>
#include <string>
#include <thread>

// Pinocchio
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/autodiff/casadi.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/spatial/explog.hpp>

// Eigen conversions
#include <eigen_conversions/eigen_kdl.h>
#include <eigen_conversions/eigen_msg.h>

// ROS related
#include <ros/ros.h>

// Eigen
#include <Eigen/Dense>

// Solver Base Class
#include <coiks/ik_solver.hpp>

// Inverse Jacobian IK Solver
#include <invj/invj_ik.hpp>

// Nonlinear Optimization IK Solver
#include <nlo/nlo_ik.hpp>

// Hierarchical Quadratic Programming IK Solver
#include <hqp/hqp_ik.hpp>

// YAML parser library
// #include "yaml-cpp/yaml.h"

// Casadi
#include <casadi/casadi.hpp>

// SolverOptions
#include <coiks/solver_options.hpp>

using namespace Eigen;
namespace pin = pinocchio;

namespace coiks
{

  class COIKS : public IkSolver
  {
  public:
    COIKS(const std::string &_urdf_file, const std::string &_base_link,
          const std::string &_tip_link, const std::string &_ik_solver,
          SolverOptions solver_opts, double _max_time = 10e-3,
          double _max_error = 1e-4, int _max_iter = 1e2, double _dt = 1.0);

    ~COIKS();

    int IkSolve(const VectorXd q_init, const pin::SE3 &x_des, VectorXd &q_out);
    int solveFk(const VectorXd q_act, pin::SE3 &x_B_Fee);

    int concurrentIkSolve(const VectorXd q_init, const pin::SE3 &x_des,
                          VectorXd &q_out);

    void printStatistics();

    int get_n_joints() { return n_joints_; }
    std::string get_solver_name() { return ik_solver_name_; }

  private:
    //   Pinocchio variables
    pin::Model model_;
    pin::Data mdl_data_;
    pin::FrameIndex ee_id_;

    std::string urdf_file_;
    std::string base_link_;
    std::string ee_link_;
    std::string ik_solver_name_;

    int n_joints_;
    int max_iter_;
    int succ_sol_tp_;
    int succ_sol_nlo_;
    int succ_sol_hqp_;
    double max_time_;
    double max_error_;
    double dt_;

    bool initialized_;

    VectorXd q_ul_;
    VectorXd q_ll_;

    std::fstream fout_;
    int prob_id_;
    int n_sol_;
    int sol_id_;

    std::vector<VectorXd> q_solutions_;
    std::vector<double> errors_;

    bool initialize();
    bool initialize_codcs();
    bool printModelInfo();

    template <typename T1, typename T2, typename T3>
    bool run3Solver(T1 &solver, T2 &other_solver1, T3 &other_solver2,
                    const VectorXd q_init, const pin::SE3 &x_des, int id);

    template <typename T1, typename T2>
    bool run2Solver(T1 &solver, T2 &other_solver1, const VectorXd q_init,
                    const pin::SE3 &x_des, int id);

    bool runINVJNLOIK(const VectorXd q_init, const pin::SE3 &x_Fee_d);
    bool runINVJHQPIK(const VectorXd q_init, const pin::SE3 &x_Fee_d);
    bool runINVJALLIK(const VectorXd q_init, const pin::SE3 &x_Fee_d);
    bool runNLOINVJIK(const VectorXd q_init, const pin::SE3 &x_Fee_d);
    bool runNLOHQPIK(const VectorXd q_init, const pin::SE3 &x_Fee_d);
    bool runNLOALLIK(const VectorXd q_init, const pin::SE3 &x_Fee_d);
    bool runHQPINVJIK(const VectorXd q_init, const pin::SE3 &x_Fee_d);
    bool runHQPNLOIK(const VectorXd q_init, const pin::SE3 &x_Fee_d);
    bool runHQPALLIK(const VectorXd q_init, const pin::SE3 &x_Fee_d);

    bool initialize_coiks_invj_nlo();
    bool initialize_coiks_invj_hqp();
    bool initialize_coiks_hqp_nlo();

    std::unique_ptr<INVJ_IkSolver<INVJ_PINOCCHIO>> invj_solver_;
    std::unique_ptr<NLO_IkSolver<NLO_CASADI>> nlo_solver_;
    std::unique_ptr<HQP_IkSolver<HQP_CASADI>> hqp_solver_;

    std::thread solver1_, solver2_, solver3_;
    std::mutex mtx_;

    std::chrono::time_point<std::chrono::high_resolution_clock>
        start_iksolve_time_;

    SolverOptions solver_opts_;
  };

  inline bool COIKS::runINVJNLOIK(const VectorXd q_init,
                                  const pin::SE3 &x_Fee_d)
  {
    return run2Solver(*invj_solver_.get(), *nlo_solver_.get(), q_init, x_Fee_d,
                      1);
  }
  inline bool COIKS::runINVJHQPIK(const VectorXd q_init,
                                  const pin::SE3 &x_Fee_d)
  {
    return run2Solver(*invj_solver_.get(), *hqp_solver_.get(), q_init, x_Fee_d,
                      1);
  }
  inline bool COIKS::runINVJALLIK(const VectorXd q_init,
                                  const pin::SE3 &x_Fee_d)
  {
    return run3Solver(*invj_solver_.get(), *nlo_solver_.get(),
                      *hqp_solver_.get(), q_init, x_Fee_d, 1);
  }

  inline bool COIKS::runNLOINVJIK(const VectorXd q_init,
                                  const pin::SE3 &x_Fee_d)
  {
    return run2Solver(*nlo_solver_.get(), *invj_solver_.get(), q_init, x_Fee_d,
                      2);
  }

  inline bool COIKS::runNLOHQPIK(const VectorXd q_init, const pin::SE3 &x_Fee_d)
  {
    return run2Solver(*nlo_solver_.get(), *hqp_solver_.get(), q_init, x_Fee_d,
                      2);
  }

  inline bool COIKS::runNLOALLIK(const VectorXd q_init, const pin::SE3 &x_Fee_d)
  {
    return run3Solver(*nlo_solver_.get(), *invj_solver_.get(),
                      *hqp_solver_.get(), q_init, x_Fee_d, 2);
  }

  inline bool COIKS::runHQPINVJIK(const VectorXd q_init,
                                  const pin::SE3 &x_Fee_d)
  {
    return run2Solver(*hqp_solver_.get(), *invj_solver_.get(), q_init, x_Fee_d,
                      3);
  }

  inline bool COIKS::runHQPNLOIK(const VectorXd q_init, const pin::SE3 &x_Fee_d)
  {
    return run2Solver(*hqp_solver_.get(), *nlo_solver_.get(), q_init, x_Fee_d,
                      3);
  }

  inline bool COIKS::runHQPALLIK(const VectorXd q_init, const pin::SE3 &x_Fee_d)
  {
    return run3Solver(*hqp_solver_.get(), *invj_solver_.get(),
                      *nlo_solver_.get(), q_init, x_Fee_d, 3);
  }

} // namespace coiks

#endif
