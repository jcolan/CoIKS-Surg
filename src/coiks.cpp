/******************************************************************************
# coiks.cpp:  COncurrent Inverse Kinematics Solver                            #
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

#include <coiks/coiks.hpp>

namespace coiks
{

  double fRand(double min, double max)
  {
    double f = (double)rand() / RAND_MAX;
    return min + f * (max - min);
  }

  COIKS::COIKS(const std::string &_urdf_file, const std::string &_base_link,
               const std::string &_ee_link, const std::string &_ik_solver,
               SolverOptions _solver_opts, double _max_time, double _max_error,
               int _max_iter, double _dt)
      : initialized_(false), max_error_(_max_error), max_time_(_max_time),
        max_iter_(_max_iter), urdf_file_(_urdf_file), base_link_(_base_link),
        ee_link_(_ee_link), ik_solver_name_(_ik_solver), dt_(_dt),
        solver_opts_(_solver_opts)
  {
    initialize();
  }

  COIKS::~COIKS()
  {
    if (solver1_.joinable())
      solver1_.join();
    if (solver2_.joinable())
      solver2_.join();
    if (solver3_.joinable())
      solver3_.join();
    if (solver_opts_.logging_enabled_)
      fout_.close();
    std::cout << "COIKS closed" << std::endl;
  }

  int COIKS::IkSolve(const VectorXd q_init, const pin::SE3 &x_des,
                     VectorXd &q_sol)
  {
    int res = 1;
    prob_id_ += 1;
    n_sol_ = 0;
    sol_id_ = 0;

    if (solver_opts_.verb_level_ >= 2)
      std::cout << "[CODCS-IK] Solving IK with " << ik_solver_name_
                << std::endl;
    start_iksolve_time_ = std::chrono::high_resolution_clock::now();
    auto start_cb_time = std::chrono::high_resolution_clock::now();

    if (ik_solver_name_ == "coiks_invj")
      res = invj_solver_->IkSolve(q_init, x_des, q_sol);
    else if (ik_solver_name_ == "coiks_nlo")
      res = nlo_solver_->IkSolve(q_init, x_des, q_sol);
    else if (ik_solver_name_ == "coiks_hqp")
      res = hqp_solver_->IkSolve(q_init, x_des, q_sol);
    else
      res = concurrentIkSolve(q_init, x_des, q_sol);

    auto stop_cb_time = std::chrono::high_resolution_clock::now();
    auto duration_cb = std::chrono::duration_cast<std::chrono::microseconds>(
        stop_cb_time - start_cb_time);
    if (solver_opts_.verb_level_ >= 2)
    {
      std::cout << "[CODCS-IK] Time IK [us]: " << duration_cb.count() << " usec"
                << std::endl;
      std::cout << "[CODCS-IK] Solution found: " << q_sol.transpose()
                << std::endl;
      std::cout << "[CODCS-IK] Solution found (deg): "
                << q_sol.transpose() * (180 / M_PI) << std::endl;
    }

    if (solver_opts_.logging_enabled_)
    {
      n_sol_ = q_solutions_.size();
      fout_ << prob_id_ << "," << duration_cb.count() << "," << n_sol_ << ","
            << sol_id_ << "," << x_des.translation()[0] << ","
            << x_des.translation()[1] << "," << x_des.translation()[2] << ","
            << x_des.rotation()(0, 0) << "," << x_des.rotation()(1, 0) << ","
            << x_des.rotation()(2, 0) << "," << x_des.rotation()(0, 1) << ","
            << x_des.rotation()(1, 1) << "," << x_des.rotation()(2, 1) << "\n";
    }

    return res;
  }
  int COIKS::solveFk(const VectorXd q_act, pin::SE3 &x_B_Fee)
  {
    int res = 1;

    pin::forwardKinematics(model_, mdl_data_, q_act);
    pin::updateFramePlacements(model_, mdl_data_);

    x_B_Fee = mdl_data_.oMf[ee_id_];
    return res;
  }

  int COIKS::concurrentIkSolve(const VectorXd q_init, const pin::SE3 &x_des,
                               VectorXd &q_sol)
  {
    int res = 1;

    q_solutions_.clear();

    if (ik_solver_name_ == "coiks_invj_nlo")
    {
      invj_solver_->reset();
      nlo_solver_->reset();

      solver1_ = std::thread(&COIKS::runINVJNLOIK, this, q_init, x_des);
      solver2_ = std::thread(&COIKS::runNLOINVJIK, this, q_init, x_des);

      solver1_.join();
      solver2_.join();
    }

    else if (ik_solver_name_ == "coiks_invj_hqp")
    {
      invj_solver_->reset();
      hqp_solver_->reset();

      solver1_ = std::thread(&COIKS::runINVJHQPIK, this, q_init, x_des);
      solver2_ = std::thread(&COIKS::runHQPINVJIK, this, q_init, x_des);

      solver1_.join();
      solver2_.join();
    }

    else if (ik_solver_name_ == "coiks_hqp_nlo")
    {
      nlo_solver_->reset();
      hqp_solver_->reset();

      solver1_ = std::thread(&COIKS::runHQPNLOIK, this, q_init, x_des);
      solver2_ = std::thread(&COIKS::runNLOHQPIK, this, q_init, x_des);

      solver1_.join();
      solver2_.join();
    }

    else if (ik_solver_name_ == "coiks_all")
    {
      invj_solver_->reset();
      nlo_solver_->reset();
      hqp_solver_->reset();

      solver1_ = std::thread(&COIKS::runINVJALLIK, this, q_init, x_des);
      solver2_ = std::thread(&COIKS::runNLOALLIK, this, q_init, x_des);
      solver3_ = std::thread(&COIKS::runHQPALLIK, this, q_init, x_des);

      solver1_.join();
      solver2_.join();
      solver3_.join();
    }
    else
    {
      std::cout << "No IK solver found" << std::endl;
      return 1;
    }

    if (!q_solutions_.empty())
    {
      q_sol = q_solutions_[0];
      if (solver_opts_.verb_level_ >= 2)
        printStatistics();
      res = 0;
    }

    return res;
  }

  void COIKS::printStatistics()
  {
    std::cout << "Solutions found: " << q_solutions_.size() << std::endl;

    std::cout << "TP"
              << " found " << succ_sol_tp_ << " solutions" << std::endl;
    std::cout << "NLO"
              << " found " << succ_sol_nlo_ << " solutions" << std::endl;
    std::cout << "HQP"
              << " found " << succ_sol_hqp_ << " solutions" << std::endl;
  }

  bool COIKS::initialize()
  {
    std::cout << "Initializing COIKS with Max. Error: " << max_error_
              << " Max. Time:" << max_time_ << " Max. It.:" << max_iter_
              << " Delta-T:" << dt_ << std::endl;
    //   Load the urdf model
    pin::urdf::buildModel(urdf_file_, model_);
    mdl_data_ = pin::Data(model_);

    n_joints_ = model_.nq;
    // q_sol_.resize(n_joints_);
    q_solutions_.clear();
    errors_.clear();
    succ_sol_tp_ = 0;
    succ_sol_nlo_ = 0;
    succ_sol_hqp_ = 0;

    // Getting Joints Limits
    q_ul_ = model_.upperPositionLimit;
    q_ll_ = model_.lowerPositionLimit;

    ee_id_ = model_.getFrameId(ee_link_);
    printModelInfo();

    if (ik_solver_name_ == "coiks_invj")
      invj_solver_.reset(new INVJ_IkSolver<INVJ_PINOCCHIO>(
          model_, ee_id_, solver_opts_, max_time_, max_error_, max_iter_, dt_));
    else if (ik_solver_name_ == "coiks_nlo")
      nlo_solver_.reset(new NLO_IkSolver<NLO_CASADI>(
          model_, ee_id_, solver_opts_, max_time_, max_error_, max_iter_, dt_));
    else if (ik_solver_name_ == "coiks_hqp")
      hqp_solver_.reset(new HQP_IkSolver<HQP_CASADI>(
          model_, ee_id_, solver_opts_, max_time_, max_error_, max_iter_, dt_));
    else if (ik_solver_name_ == "coiks_invj_nlo")
      initialize_coiks_invj_nlo();
    else if (ik_solver_name_ == "coiks_invj_hqp")
      initialize_coiks_invj_hqp();
    else if (ik_solver_name_ == "coiks_hqp_nlo")
      initialize_coiks_hqp_nlo();
    else if (ik_solver_name_ == "coiks_all")
      initialize_codcs();
    else
    {
      std::cout << "No IK solver found. Using default: COIKS-INVJ" << std::endl;
      invj_solver_.reset(new INVJ_IkSolver<INVJ_PINOCCHIO>(
          model_, ee_id_, solver_opts_, max_time_, max_error_, max_iter_, dt_));
    }

    std::cout << "COIKS initialized" << std::endl;
    initialized_ = true;

    if (solver_opts_.logging_enabled_)
    {
      prob_id_ = 0;
      n_sol_;
      sol_id_;

      time_t currentTime;
      struct tm *localTime;

      time(&currentTime); // Get the current time
      localTime = localtime(&currentTime);

      fout_.open(solver_opts_.log_path_ + "log_concurrent_" +
                     std::to_string(localTime->tm_mday) +
                     std::to_string(localTime->tm_hour) +
                     std::to_string(localTime->tm_min) +
                     std::to_string(localTime->tm_sec) + ".csv",
                 std::ios::out | std::ios::app);
      fout_ << "idx,time,n_sol,sol_id,p_x,p_y,"
               "p_z,r_11,r_21,r_31,r_21,r_22,r_23 \n";
    }

    return true;
  }

  bool COIKS::initialize_coiks_invj_nlo()
  {
    invj_solver_.reset(new INVJ_IkSolver<INVJ_PINOCCHIO>(
        model_, ee_id_, solver_opts_, max_time_, max_error_, max_iter_, dt_));
    nlo_solver_.reset(new NLO_IkSolver<NLO_CASADI>(
        model_, ee_id_, solver_opts_, max_time_, max_error_, max_iter_, dt_));
    return true;
  }

  bool COIKS::initialize_coiks_invj_hqp()
  {
    invj_solver_.reset(new INVJ_IkSolver<INVJ_PINOCCHIO>(
        model_, ee_id_, solver_opts_, max_time_, max_error_, max_iter_, dt_));
    hqp_solver_.reset(new HQP_IkSolver<HQP_CASADI>(
        model_, ee_id_, solver_opts_, max_time_, max_error_, max_iter_, dt_));

    return true;
  }

  bool COIKS::initialize_coiks_hqp_nlo()
  {
    nlo_solver_.reset(new NLO_IkSolver<NLO_CASADI>(
        model_, ee_id_, solver_opts_, max_time_, max_error_, max_iter_, dt_));
    hqp_solver_.reset(new HQP_IkSolver<HQP_CASADI>(
        model_, ee_id_, solver_opts_, max_time_, max_error_, max_iter_, dt_));

    return true;
  }

  bool COIKS::initialize_codcs()
  {
    invj_solver_.reset(new INVJ_IkSolver<INVJ_PINOCCHIO>(
        model_, ee_id_, solver_opts_, max_time_, max_error_, max_iter_, dt_));
    nlo_solver_.reset(new NLO_IkSolver<NLO_CASADI>(
        model_, ee_id_, solver_opts_, max_time_, max_error_, max_iter_, dt_));
    hqp_solver_.reset(new HQP_IkSolver<HQP_CASADI>(
        model_, ee_id_, solver_opts_, max_time_, max_error_, max_iter_, dt_));

    return true;
  }

  bool COIKS::printModelInfo()
  {
    std::cout << "\nPrinting Model Info \n-----------------------" << std::endl;
    std::cout << "Number of Joints found in model: " << model_.njoints << "\n";
    std::cout << "Model nq (positon states): " << model_.nq << "\n";
    std::cout << "Model nv (velocity states): " << model_.nv << "\n";
    std::cout << "Joints lower limits [rad]: "
              << model_.lowerPositionLimit.transpose() << "\n";
    std::cout << "Joints upper limits [rad]: "
              << model_.upperPositionLimit.transpose() << "\n";
    std::cout << "EE link name: " << ee_link_ << std::endl;
    std::cout << "EE link frame id: " << ee_id_ << std::endl;
    return true;
  }

  template <typename T1, typename T2>
  bool COIKS::run2Solver(T1 &solver, T2 &other_solver1, const VectorXd q_init,
                         const pin::SE3 &x_des, int id)
  {
    VectorXd q_sol;
    double time_left;

    std::chrono::microseconds diff;
    std::chrono::microseconds diff_solver;

    while (true)
    {
      diff = std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::high_resolution_clock::now() - start_iksolve_time_);
      time_left = max_time_ - diff.count() / 1000000.0;

      if (time_left <= 0)
        break;

      solver.set_max_time(time_left);

      bool res = solver.IkSolve(q_init, x_des, q_sol);

      mtx_.lock();

      if (!res)
      {
        if (id == 1)
        {
          succ_sol_tp_++;
          if (q_solutions_.empty())
            sol_id_ = 1;
        }
        else if (id == 2)
        {
          succ_sol_nlo_++;
          if (q_solutions_.empty())
            sol_id_ = 2;
        }
        else if (id == 3)
        {
          succ_sol_hqp_++;
          if (q_solutions_.empty())
            sol_id_ = 3;
        }
        q_solutions_.push_back(q_sol);
      }
      mtx_.unlock();

      if (!q_solutions_.empty())
      {
        break;
      }
    }

    other_solver1.abort();
    solver.set_max_time(max_time_);

    return true;
  }

  template <typename T1, typename T2, typename T3>
  bool COIKS::run3Solver(T1 &solver, T2 &other_solver1, T3 &other_solver2,
                         const VectorXd q_init, const pin::SE3 &x_des, int id)
  {
    VectorXd q_sol;
    double time_left;

    std::chrono::microseconds diff;
    std::chrono::microseconds diff_solver;

    while (true)
    {
      diff = std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::high_resolution_clock::now() - start_iksolve_time_);
      time_left = max_time_ - diff.count() / 1000000.0;

      if (time_left <= 0)
        break;

      solver.set_max_time(time_left);

      bool res = solver.IkSolve(q_init, x_des, q_sol);

      mtx_.lock();

      if (!res)
      {
        if (id == 1)
        {
          succ_sol_tp_++;
          if (q_solutions_.empty())
            sol_id_ = 1;
        }
        else if (id == 2)
        {
          succ_sol_nlo_++;
          if (q_solutions_.empty())
            sol_id_ = 2;
        }
        else if (id == 3)
        {
          succ_sol_hqp_++;
          if (q_solutions_.empty())
            sol_id_ = 3;
        }
        q_solutions_.push_back(q_sol);
      }
      mtx_.unlock();

      if (!q_solutions_.empty())
      {
        break;
      }
    }

    other_solver1.abort();
    other_solver2.abort();
    solver.set_max_time(max_time_);

    return true;
  }

} // namespace coiks