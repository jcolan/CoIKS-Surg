/******************************************************************************
# hqp_casadi.hpp:  CoIKS HQP IK solver                                           #
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

#ifndef HQP_CASADI_HPP
#define HQP_CASADI_HPP

#include <coiks/solver_base.hpp>
#include <coiks/solver_options.hpp>
// Casadi
#include <casadi/casadi.hpp>
// C++
#include <map>
// Pinocchio
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/autodiff/casadi.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/spatial/explog.hpp>

namespace ca = casadi;

namespace coiks
{

  class HQP_CASADI : public SolverBase
  {
  private:
    pin::Model mdl_;
    pin::Data mdl_data_;
    pin::FrameIndex id_Fee_;
    pin::FrameIndex id_Fprercm_;
    pin::FrameIndex id_Fpostrcm_;

    int max_iter_;
    double max_time_;
    double max_error_;
    double dt_;
    int n_q_;

    double hqp_Ke1_;
    double hqp_Ke2_;
    double hqp_Kd1_;
    double hqp_Kd2_;

    int verb_level_;
    std::string time_stats_;

    VectorXd q_ul_;
    VectorXd q_ll_;
    VectorXd q_sol_;
    std::vector<double> q_sol_vec_;

    Vector3d B_p_Ftrocar_;

    bool aborted_;
    bool success_;
    bool constrained_control_;
    bool manipulability_enabled_;
    bool logging_enabled_;

    std::string log_path_;

    ca::Function qpsolver1_;
    ca::Function qpsolver2_;

    MatrixXd C_eig_;
    MatrixXd d_eig_;
    // MatrixXd d_ext;

    //* Task 1
    MatrixXd A_1_;
    MatrixXd b_1_;
    MatrixXd C_1_;
    MatrixXd d_1_;
    MatrixXd Abar_1_;
    MatrixXd bbar_1_;
    MatrixXd Cbar_1_;
    MatrixXd Q_1_;
    MatrixXd p_1_;

    MatrixXd param1;
    ca::DM par1;

    ca::SXDict qp_1;

    //* Task 2
    MatrixXd N_1_;
    MatrixXd A_2_;
    MatrixXd b_2_;
    MatrixXd C_2_;
    MatrixXd d_2_;
    MatrixXd Abar_2_;
    MatrixXd bbar_2_;
    MatrixXd Cbar_2_;
    MatrixXd Q_2_;
    MatrixXd p_2_;
    MatrixXd A_t2_;
    MatrixXd b_t2_;

    MatrixXd param2;
    ca::DM par2;

    ca::SXDict qp_2;

    // Task 3
    MatrixXd A_3_;
    MatrixXd b_3_;

    double rcm_error_max_;
    std::string error_type_;
    std::string prercm_joint_name_;
    std::string postrcm_joint_name_;

    bool hqp_warm_start_;

    std::vector<double> cost_coeff_;

    SolverOptions solver_opts_;

    // logger output
    std::fstream fout_;
    int prob_id_;

  public:
    HQP_CASADI(const pin::Model &_model, const pin::FrameIndex &_Fee_id,
               SolverOptions _solver_opts, const double _max_time,
               const double _max_error, const int _max_iter, const double _dt);
    ~HQP_CASADI();

    int IkSolve(const VectorXd q_init, const pin::SE3 &x_d, VectorXd &q_sol);
    int computeEETaskError(const pin::SE3 &B_x_Fee, const pin::SE3 &x_d,
                           Vector6d &err_ee);
    int computeEETaskError(const pin::SE3 &B_x_Fee, const pin::SE3 &x_d,
                           const std::string err_type, Vector6d &err_ee);
    int computeRCMTaskError(const pin::SE3 &B_x_Fprercm,
                            const pin::SE3 &B_x_Fpostrcm,
                            const MatrixXd &B_Jb_Fprercm,
                            const MatrixXd &B_Jb_Fpostrcm, double &err_rcm,
                            VectorXd &B_Jb_Frcm);
    ca::DM eig_to_casDM(const VectorXd &eig);
    ca::DM eigmat_to_casDM(const MatrixXd &eig);
    MatrixXd casDM_to_eig(const casadi::DM &cas);
    double manipEE(pin::Model model, pin::Data data, VectorXd q,
                   pin::FrameIndex ee_id);
    VectorXd gradManipEE(pin::Model model, pin::Data data, VectorXd q,
                         pin::FrameIndex ee_id);

    MatrixXd pseudoInverse(const Eigen::MatrixXd &a, double epsilon);

    void generate_qp_solver();
    void GetOptions(SolverOptions _solver_opts);

    inline void abort() { aborted_ = true; }
    inline void reset() { aborted_ = false; }
    inline void set_max_time(double _max_time) { max_time_ = _max_time; }
    inline void set_max_error(double _max_error) { max_error_ = _max_error; }
  };

  void HQP_CASADI::GetOptions(SolverOptions _solver_opts)
  {
    SolverOptions so;

    so = _solver_opts;

    // Solver parameters
    error_type_ = so.error_type_;
    manipulability_enabled_ = so.manipulability_enabled_;
    hqp_Kd1_ = so.hqp_Kd1_;
    hqp_Kd2_ = so.hqp_Kd2_;
    hqp_Ke1_ = so.hqp_Ke1_;
    hqp_Ke2_ = so.hqp_Ke2_;
    hqp_warm_start_ = so.hqp_warm_start_;

    // Constrained control
    constrained_control_ = so.constrained_control_;
    B_p_Ftrocar_ = so.trocar_pos_;
    rcm_error_max_ = so.rcm_error_max_;
    prercm_joint_name_ = so.prercm_joint_name_;
    postrcm_joint_name_ = so.postrcm_joint_name_;

    // Logging
    time_stats_ = so.time_stats_;
    verb_level_ = so.verb_level_;
    logging_enabled_ = so.logging_enabled_;
    log_path_ = so.log_path_;

    std::cout << "\n------\nHQP Options summary:" << std::endl;
    std::cout << "Constrained control enabled: " << constrained_control_
              << std::endl;
    std::cout << "Manipulability enabled: " << manipulability_enabled_
              << std::endl;
    std::cout << "Error computation type: " << error_type_ << std::endl;
    std::cout << "Max. EE error: " << max_error_ << std::endl;
    std::cout << "Verbosity Level: " << verb_level_ << std::endl;
    std::cout << "Time Statstics: " << time_stats_ << std::endl;
    std::cout << "Logging enabled: " << logging_enabled_ << std::endl;
    std::cout << "hqp_Ke1 : " << hqp_Ke1_ << std::endl;
    std::cout << "hqp_Ke2 : " << hqp_Ke2_ << std::endl;
    std::cout << "hqp_Kd1 : " << hqp_Kd1_ << std::endl;
    std::cout << "hqp_Kd2 : " << hqp_Kd2_ << std::endl;
    std::cout << "HQP warm start : " << hqp_warm_start_ << std::endl;

    if (constrained_control_)
    {
      std::cout << "Max. RCM error: " << rcm_error_max_ << std::endl;
      std::cout << "Pre-RCM joint: " << prercm_joint_name_ << std::endl;
      std::cout << "Post-RCM joint: " << postrcm_joint_name_ << std::endl;
      std::cout << "Trocar Position: " << B_p_Ftrocar_.transpose() << std::endl;
    }
  }

  HQP_CASADI::HQP_CASADI(const pin::Model &_model,
                         const pin::FrameIndex &_Fee_id,
                         SolverOptions _solver_opts, const double _max_time,
                         const double _max_error, const int _max_iter,
                         const double _dt)
      : aborted_(false), success_(false), mdl_(_model), n_q_(_model.nq),
        id_Fee_(_Fee_id), max_time_(_max_time), max_error_(_max_error),
        max_iter_(_max_iter), dt_(_dt)
  {
    std::cout << "----------\nInitializing IK solver COIKS-HQP" << std::endl;
    GetOptions(_solver_opts);

    mdl_data_ = pin::Data(mdl_);

    if (constrained_control_)
    {
      id_Fpostrcm_ = mdl_.getFrameId(postrcm_joint_name_);
      id_Fprercm_ = mdl_.getFrameId(prercm_joint_name_);
    }

    q_ul_ = mdl_.upperPositionLimit;
    q_ll_ = mdl_.lowerPositionLimit;

    std::cout << "Generating QP solver " << std::endl;
    generate_qp_solver();

    C_eig_.resize(2 * n_q_, n_q_);
    C_eig_.setZero();
    C_eig_ << MatrixXd::Identity(n_q_, n_q_), -MatrixXd::Identity(n_q_, n_q_);

    d_eig_.resize(2 * n_q_, 1);
    d_eig_.setZero();

    //* Task 1
    A_1_.resize(1, n_q_);
    A_1_.setZero();

    b_1_.resize(1, 1);
    b_1_.setZero();

    C_1_ = C_eig_;

    d_1_ = d_eig_;
    d_1_.setZero();

    Abar_1_.resize(2 * n_q_ + 1, 3 * n_q_);
    Abar_1_.setZero();

    bbar_1_.resize(2 * n_q_ + 1, 1);
    bbar_1_.setZero();

    Cbar_1_.resize(2 * n_q_, 3 * n_q_);
    Cbar_1_.setZero();

    Q_1_.resize(3 * n_q_, 3 * n_q_);
    Q_1_.setZero();

    p_1_.resize(3 * n_q_, 1);
    p_1_.setZero();

    param1.resize(3 * n_q_, 5 * n_q_ + 2);
    param1.setZero();
    par1.resize(3 * n_q_, 5 * n_q_ + 2);

    //* Task 2
    N_1_.resize(n_q_, n_q_);
    N_1_.setZero();

    if (error_type_ == "only_p")
    {
      A_2_.resize(3, n_q_);
      b_2_.resize(3, 1);
    }
    else
    {
      A_2_.resize(6, n_q_);
      b_2_.resize(6, 1);
    }

    A_2_.setZero();
    b_2_.setZero();

    C_2_ = C_eig_;
    d_2_ = d_eig_;

    if (constrained_control_)
    {
      if (error_type_ == "only_p")
      {
        if (manipulability_enabled_)
        {
          Abar_2_.resize(2 * n_q_ + 4, 3 * n_q_);
          bbar_2_.resize(2 * n_q_ + 4, 1);
        }
        else
        {
          Abar_2_.resize(2 * n_q_ + 3, 3 * n_q_);
          bbar_2_.resize(2 * n_q_ + 3, 1);
        }
      }
      else
      {
        if (manipulability_enabled_)
        {

          Abar_2_.resize(2 * n_q_ + 7, 3 * n_q_);
          bbar_2_.resize(2 * n_q_ + 7, 1);
        }
        else
        {
          Abar_2_.resize(2 * n_q_ + 6, 3 * n_q_);
          bbar_2_.resize(2 * n_q_ + 6, 1);
        }
      }
    }
    else
    {
      Abar_2_.resize(2 * n_q_ + 6, 3 * n_q_);
      bbar_2_.resize(2 * n_q_ + 6, 1);
    }

    Abar_2_.setZero();
    bbar_2_.setZero();

    Cbar_2_.resize(2 * n_q_, 3 * n_q_);
    Cbar_2_.setZero();
    Q_2_.resize(3 * n_q_, 3 * n_q_);
    Q_2_.setZero();
    p_2_.resize(3 * n_q_, 1);
    p_2_.setZero();

    if (error_type_ == "only_p")
    {
      A_t2_.resize(4, n_q_);
      b_t2_.resize(4, 1);
    }
    else
    {
      A_t2_.resize(7, n_q_);
      b_t2_.resize(7, 1);
    }

    A_t2_.setZero();
    b_t2_.setZero();

    param2.resize(3 * n_q_, 5 * n_q_ + 2);
    par2.resize(3 * n_q_, 5 * n_q_ + 2);

    // Task 3
    A_3_.resize(1, n_q_);
    b_3_.resize(1, 1);

    // Getting Joints
    std::cout << "Joints lower limits: " << mdl_.lowerPositionLimit.transpose()
              << std::endl;
    std::cout << "Joints upper limits: " << mdl_.upperPositionLimit.transpose()
              << std::endl;
    std::cout << "Joints considered for IK solver: " << n_q_ << std::endl;

    prob_id_ = 0;

    if (logging_enabled_)
    {
      time_t currentTime;
      struct tm *localTime;

      time(&currentTime); // Get the current time
      localTime = localtime(&currentTime);

      fout_.open(log_path_ + "log_hqp_" + std::to_string(localTime->tm_mday) +
                     std::to_string(localTime->tm_hour) +
                     std::to_string(localTime->tm_min) +
                     std::to_string(localTime->tm_sec) + ".csv",
                 std::ios::out | std::ios::app);
      fout_ << "idx,time,error_rcm,error_ee,error_t,error_r,p_x,p_y,p_z,r_11,r_"
               "21,r_31,r_12,r_22,r_32\n";
    }
  }

  HQP_CASADI::~HQP_CASADI()
  {
    if (logging_enabled_)
      fout_.close();
  }

  int HQP_CASADI::computeEETaskError(const pin::SE3 &B_x_Fee,
                                     const pin::SE3 &x_d, Vector6d &err_ee)
  {

    //* Computation Errors
    if (error_type_ == "log6")
    {
      //? Using log6
      pin::SE3 Fee_Hact_B(B_x_Fee.rotation().transpose(),
                          -B_x_Fee.rotation().transpose() *
                              B_x_Fee.translation());
      pin::SE3 B_Hdes_Fee(x_d);
      pin::SE3 B_Herr(B_Hdes_Fee.act(Fee_Hact_B));

      err_ee = pin::log6(B_Herr).toVector();
    }
    else if (error_type_ == "log3")
    {
      //? Using log3
      Vector3d err_tr = x_d.translation() - B_x_Fee.translation();
      Vector3d err_rot =
          pin::log3(x_d.rotation() * B_x_Fee.rotation().transpose());
      err_ee << err_tr, err_rot;
    }
    else
    {
      //? Using only p error
      Vector3d err_tr = (x_d.translation() - B_x_Fee.translation());
      Vector3d err_rot = Vector3d::Zero();
      err_ee << err_tr, err_rot;
    }

    return 0;
  }

  int HQP_CASADI::computeEETaskError(const pin::SE3 &B_x_Fee,
                                     const pin::SE3 &x_d,
                                     const std::string err_type,
                                     Vector6d &err_ee)
  {

    //* Computation Errors
    if (err_type == "log6")
    {
      //? Using log6
      pin::SE3 Fee_Hact_B(B_x_Fee.rotation().transpose(),
                          -B_x_Fee.rotation().transpose() *
                              B_x_Fee.translation());
      pin::SE3 B_Hdes_Fee(x_d);
      pin::SE3 B_Herr(B_Hdes_Fee.act(Fee_Hact_B));

      err_ee = pin::log6(B_Herr).toVector();
    }
    else if (err_type == "log3")
    {
      //? Using log3
      Vector3d err_tr = x_d.translation() - B_x_Fee.translation();
      Vector3d err_rot =
          pin::log3(x_d.rotation() * B_x_Fee.rotation().transpose());
      err_ee << err_tr, err_rot;
    }
    else
    {
      //? Using only p error
      Vector3d err_tr = (x_d.translation() - B_x_Fee.translation());
      Vector3d err_rot = Vector3d::Zero();
      err_ee << err_tr, err_rot;
      // err3d_ee = err_tr;
    }
    return 0;
  }
  int HQP_CASADI::computeRCMTaskError(const pin::SE3 &B_x_Fprercm,
                                      const pin::SE3 &B_x_Fpostrcm,
                                      const MatrixXd &B_Jb_Fprercm,
                                      const MatrixXd &B_Jb_Fpostrcm,
                                      double &err_rcm, VectorXd &B_Jb_Frcm)
  {
    //* Computing RCM error
    Vector3d ps = B_x_Fpostrcm.translation() - B_x_Fprercm.translation();
    Vector3d pr = B_p_Ftrocar_ - B_x_Fprercm.translation();
    Vector3d ps_hat = ps.normalized(); //
    Vector3d pr_hat = pr.normalized(); //

    //* Compute RCM Pose B_x_Frcm
    Vector3d B_p_Frcm;
    B_p_Frcm = B_x_Fprercm.translation() + pr.transpose() * ps_hat * ps_hat;

    pin::SE3 B_x_Frcm(Matrix3d::Identity(), B_p_Frcm);

    //* Computation of RCM Error
    Vector3d pe = B_p_Ftrocar_ - B_x_Frcm.translation();
    Vector3d pe_hat = pe.normalized();

    err_rcm = pe.norm();

    if (verb_level_ >= 1)
      std::cout << "error RCM: " << err_rcm << " -> " << pe.transpose()
                << std::endl;

    //*  Compute B_Jb_Frcm_
    pin::Data::Matrix3x Jb_ps_hat =
        (1 / ps.norm()) *
        (Matrix3d::Identity() -
         (1 / (ps.transpose() * ps)) * ps_hat * ps_hat.transpose()) *
        (B_Jb_Fpostrcm - B_Jb_Fprercm).topRows(3);

    B_Jb_Frcm = pe_hat.transpose() *
                ((Matrix3d::Identity() - ps_hat * ps_hat.transpose()) *
                     B_Jb_Fprercm.topRows(3) +
                 (ps_hat * pr.transpose() +
                  pr.transpose() * ps_hat * Matrix3d::Identity()) *
                     Jb_ps_hat);

    return 0;
  }

  int HQP_CASADI::IkSolve(const VectorXd q_init, const pin::SE3 &x_d,
                          VectorXd &q_sol)
  {
    std::chrono::time_point<std::chrono::high_resolution_clock>
        start_solve_time = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds diff;

    auto start_cb_time = std::chrono::high_resolution_clock::now();

    prob_id_ += 1;

    double time_left;

    q_sol = q_init;
    VectorXd q_it = q_init;

    Vector6d err_ee;
    Vector6d err_log3_ee;
    double err_rcm = 0.0;
    err_ee.setZero();

    //? Variables for constrained Control
    pin::Data::Matrix6x B_Jb_Fee(6, n_q_);
    pin::Data::Matrix6x B_Jb_Fprercm(6, n_q_);
    pin::Data::Matrix6x B_Jb_Fpostrcm(6, n_q_);
    VectorXd B_Jb_Frcm(1, n_q_);

    B_Jb_Fee.setZero();
    B_Jb_Fprercm.setZero();
    B_Jb_Fpostrcm.setZero();
    B_Jb_Frcm.setZero();

    MatrixXd qd_opt;

    ca::DM qd_opt_1;
    ca::DM qd_opt_2;

    double mp = 0.0;
    VectorXd grad_mp;
    success_ = false;
    aborted_ = false;

    pin::SE3 B_x_Fee;
    pin::SE3 B_x_Fprercm;
    pin::SE3 B_x_Fpostrcm;

    pin::framesForwardKinematics(mdl_, mdl_data_, q_it);

    if (constrained_control_)
    {
      //* For Task Priority 1
      B_x_Fprercm = mdl_data_.oMf[id_Fprercm_];
      B_x_Fpostrcm = mdl_data_.oMf[id_Fpostrcm_];

      pin::computeFrameJacobian(mdl_, mdl_data_, q_it, id_Fprercm_,
                                pin::ReferenceFrame::LOCAL_WORLD_ALIGNED,
                                B_Jb_Fprercm);
      pin::computeFrameJacobian(mdl_, mdl_data_, q_it, id_Fpostrcm_,
                                pin::ReferenceFrame::LOCAL_WORLD_ALIGNED,
                                B_Jb_Fpostrcm);

      computeRCMTaskError(B_x_Fprercm, B_x_Fpostrcm, B_Jb_Fprercm,
                          B_Jb_Fpostrcm, err_rcm, B_Jb_Frcm);
      if (verb_level_ >= 1)
        std::cout << "[HQP] Initial error RCM: " << err_rcm << std::endl;
    }

    //* For Task Priority 2
    B_x_Fee = mdl_data_.oMf[id_Fee_];

    //* Computing EE error
    computeEETaskError(B_x_Fee, x_d, err_ee);
    if (verb_level_ >= 1)
      std::cout << "[HQP] Initial EE error: " << err_ee.norm() << " -> "
                << err_ee.transpose() << std::endl;

    //* Computing EE Jacobian
    pin::computeFrameJacobian(mdl_, mdl_data_, q_it, id_Fee_,
                              pin::ReferenceFrame::WORLD, B_Jb_Fee);

    //* For Task Priority 3: Manipulability
    if (manipulability_enabled_)
    {
      // Computing manipulability
      mp = manipEE(mdl_, mdl_data_, q_it, id_Fee_);
      if (verb_level_ >= 1)
        std::cout << "[HQP] Initial manipulability index: " << mp << std::endl;

      // Computing gradient
      grad_mp = gradManipEE(mdl_, mdl_data_, q_it, id_Fee_);
      if (verb_level_ >= 1)
        std::cout << "[HQP] Initial GRAD manipulability index: "
                  << grad_mp.transpose() << std::endl;
    }

    ca::DMDict arg_qpopt = {{"lbg", 0}};

    start_cb_time = std::chrono::high_resolution_clock::now();

    diff = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start_solve_time);
    time_left = max_time_ - diff.count() / 1000000.0;

    for (int it = 0;; it++)
    {
      d_eig_.block(0, 0, n_q_, 1) = q_ul_ - q_it;
      d_eig_.block(n_q_, 0, n_q_, 1) = -(q_ll_ - q_it);

      if (constrained_control_)
      {
        A_1_ = B_Jb_Frcm.transpose();
        b_1_(0, 0) = err_rcm;
        C_1_ = C_eig_;
        d_1_ = d_eig_;

        Abar_1_.block(0, 0, 1, n_q_) = A_1_;
        Abar_1_.block(1, n_q_, 2 * n_q_, 2 * n_q_) =
            MatrixXd::Identity(2 * n_q_, 2 * n_q_);

        bbar_1_(0, 0) = b_1_(0, 0);
        bbar_1_.block(1, 0, 2 * n_q_, 1) = MatrixXd::Zero(2 * n_q_, 1);

        Cbar_1_.block(0, 0, 2 * n_q_, n_q_) = C_1_;
        Cbar_1_.block(0, n_q_, 2 * n_q_, 2 * n_q_) =
            -MatrixXd::Identity(2 * n_q_, 2 * n_q_);

        Q_1_ = Abar_1_.transpose() * Abar_1_;
        p_1_ = -Abar_1_.transpose() * bbar_1_;

        param1.block(0, 0, 3 * n_q_, 3 * n_q_) = Q_1_;
        param1.block(0, 3 * n_q_, 3 * n_q_, 1) = p_1_;
        param1.block(0, 3 * n_q_ + 1, 3 * n_q_, 2 * n_q_) = Cbar_1_.transpose();
        param1.block(0, 5 * n_q_ + 1, 2 * n_q_, 1) = d_1_;

        par1 = eigmat_to_casDM(param1);

        ca::DMDict arg_qpopt = {{"p", par1}, {"lbg", 0}};

        ca::DMDict res_qpsolver1_ = qpsolver1_(arg_qpopt);

        qd_opt_1 = res_qpsolver1_["x"](ca::Slice(0, n_q_));
        if (verb_level_ >= 1)
          std::cout << "Q Delta-SOLUTION 1 : " << qd_opt_1 << std::endl;

        MatrixXd qd_opt_hat_1_tmp = casDM_to_eig(qd_opt_1);

        N_1_ =
            MatrixXd::Identity(n_q_, n_q_) - pseudoInverse(A_1_, 1e-10) * A_1_;

        if (error_type_ == "only_p")
        {
          A_2_ = B_Jb_Fee.topRows(3);
          b_2_ = 1.0 * err_ee.head(3);
        }
        else
        {
          A_2_ = B_Jb_Fee;
          b_2_ = 1.0 * err_ee;
        }

        //?Manipulability Maximization enabled
        if (manipulability_enabled_)
        {
          A_3_ = 0.01 * dt_ * grad_mp.transpose();
          b_3_(0, 0) = 0.025 * mp;
          C_2_ = C_eig_;
          d_2_ = d_eig_;

          if (error_type_ == "only_p")
          {
            A_t2_.block(0, 0, 3, n_q_) = A_2_;
            A_t2_.block(3, 0, 1, n_q_) = A_3_;
            b_t2_.block(0, 0, 3, 1) = b_2_;
            b_t2_.block(3, 0, 1, 1) = b_3_;
            Abar_2_.block(0, 0, 4, n_q_) = A_t2_ * N_1_;
            Abar_2_.block(4, n_q_, 2 * n_q_, 2 * n_q_) =
                MatrixXd::Identity(2 * n_q_, 2 * n_q_);

            bbar_2_.block(0, 0, 4, 1) = -(A_t2_ * qd_opt_hat_1_tmp - b_t2_);
            bbar_2_.block(4, 0, 2 * n_q_, 1) = MatrixXd::Zero(2 * n_q_, 1);
          }
          else
          {
            A_t2_.block(0, 0, 6, n_q_) = A_2_;
            A_t2_.block(6, 0, 1, n_q_) = A_3_;
            b_t2_.block(0, 0, 6, 1) = b_2_;
            b_t2_.block(6, 0, 1, 1) = b_3_;
            Abar_2_.block(0, 0, 7, n_q_) = A_t2_ * N_1_;
            Abar_2_.block(7, n_q_, 2 * n_q_, 2 * n_q_) =
                MatrixXd::Identity(2 * n_q_, 2 * n_q_);

            bbar_2_.block(0, 0, 7, 1) = -(A_t2_ * qd_opt_hat_1_tmp - b_t2_);
            bbar_2_.block(7, 0, 2 * n_q_, 1) = MatrixXd::Zero(2 * n_q_, 1);
          }

          Cbar_2_.block(0, 0, 2 * n_q_, n_q_) = C_2_ * N_1_;
          Cbar_2_.block(0, n_q_, 2 * n_q_, 2 * n_q_) =
              -MatrixXd::Identity(2 * n_q_, 2 * n_q_);

          Q_2_ = Abar_2_.transpose() * Abar_2_;
          p_2_ = -Abar_2_.transpose() * bbar_2_;

          param2.block(0, 0, 3 * n_q_, 3 * n_q_) = Q_2_;
          param2.block(0, 3 * n_q_, 3 * n_q_, 1) = p_2_;
          param2.block(0, 3 * n_q_ + 1, 3 * n_q_, 2 * n_q_) =
              Cbar_2_.transpose();
          param2.block(0, 5 * n_q_ + 1, 2 * n_q_, 1) = d_2_;

          par2 = eigmat_to_casDM(param2);

          arg_qpopt = {{"p", par2}, {"lbg", 0}};

          ca::DMDict res_qpsolver2_ = qpsolver2_(arg_qpopt);

          qd_opt_2 = res_qpsolver2_["x"](ca::Slice(0, n_q_));
          if (verb_level_ >= 1)
            std::cout << "Q Delta-SOLUTION 2: " << qd_opt_2 << std::endl;
        }
        //?Manipulability Maximization disabled
        else
        {
          C_2_ = C_eig_;
          d_2_ = d_eig_;

          if (error_type_ == "only_p")
          {
            Abar_2_.block(0, 0, 3, n_q_) = A_2_ * N_1_;
            Abar_2_.block(3, n_q_, 2 * n_q_, 2 * n_q_) =
                MatrixXd::Identity(2 * n_q_, 2 * n_q_);

            bbar_2_.block(0, 0, 3, 1) = -(A_2_ * qd_opt_hat_1_tmp - b_2_);
            bbar_2_.block(3, 0, 2 * n_q_, 1) = MatrixXd::Zero(2 * n_q_, 1);
          }
          else
          {
            Abar_2_.block(0, 0, 6, n_q_) = A_2_ * N_1_;
            Abar_2_.block(6, n_q_, 2 * n_q_, 2 * n_q_) =
                MatrixXd::Identity(2 * n_q_, 2 * n_q_);

            bbar_2_.block(0, 0, 6, 1) = -(A_2_ * qd_opt_hat_1_tmp - b_2_);
            bbar_2_.block(6, 0, 2 * n_q_, 1) = MatrixXd::Zero(2 * n_q_, 1);
          }

          Cbar_2_.block(0, 0, 2 * n_q_, n_q_) = C_2_ * N_1_;
          Cbar_2_.block(0, n_q_, 2 * n_q_, 2 * n_q_) =
              -MatrixXd::Identity(2 * n_q_, 2 * n_q_);

          Q_2_ = Abar_2_.transpose() * Abar_2_;
          p_2_ = -Abar_2_.transpose() * bbar_2_;

          param2.block(0, 0, 3 * n_q_, 3 * n_q_) = Q_2_;
          param2.block(0, 3 * n_q_, 3 * n_q_, 1) = p_2_;
          param2.block(0, 3 * n_q_ + 1, 3 * n_q_, 2 * n_q_) =
              Cbar_2_.transpose();
          param2.block(0, 5 * n_q_ + 1, 2 * n_q_, 1) = d_2_;

          par2 = eigmat_to_casDM(param2);

          arg_qpopt = {{"p", par2}, {"lbg", 0}};

          ca::DMDict res_qpsolver2_ = qpsolver2_(arg_qpopt);

          qd_opt_2 = res_qpsolver2_["x"](ca::Slice(0, n_q_));

          if (verb_level_ >= 1)
            std::cout << "Q Delta-SOLUTION 2: " << qd_opt_2 << std::endl;
        }

        MatrixXd qd_opt_hat_2_tmp =
            N_1_ * casDM_to_eig(qd_opt_2) + qd_opt_hat_1_tmp;

        qd_opt = qd_opt_hat_2_tmp;
      }
      else
      {

        A_2_ = B_Jb_Fee;
        b_2_ = 1.0 * err_ee;
        C_2_ = C_eig_;
        d_2_ = d_eig_;

        Abar_2_.block(0, 0, 6, n_q_) = A_2_;
        Abar_2_.block(6, n_q_, 2 * n_q_, 2 * n_q_) =
            MatrixXd::Identity(2 * n_q_, 2 * n_q_);

        bbar_2_.block(0, 0, 6, 1) = b_2_;
        bbar_2_.block(6, 0, 2 * n_q_, 1) = MatrixXd::Zero(2 * n_q_, 1);

        Cbar_2_.block(0, 0, 2 * n_q_, n_q_) = C_2_;
        Cbar_2_.block(0, n_q_, 2 * n_q_, 2 * n_q_) =
            -MatrixXd::Identity(2 * n_q_, 2 * n_q_);

        Q_2_ = Abar_2_.transpose() * Abar_2_;
        p_2_ = -Abar_2_.transpose() * bbar_2_;

        param2.block(0, 0, 3 * n_q_, 3 * n_q_) = Q_2_;
        param2.block(0, 3 * n_q_, 3 * n_q_, 1) = p_2_;
        param2.block(0, 3 * n_q_ + 1, 3 * n_q_, 2 * n_q_) = Cbar_2_.transpose();
        param2.block(0, 5 * n_q_ + 1, 2 * n_q_, 1) = d_2_;

        par2 = eigmat_to_casDM(param2);

        ca::DMDict arg_qpopt = {{"p", par2}, {"lbg", 0}};

        ca::DMDict res_qpsolver2_ = qpsolver1_(arg_qpopt);
        ca::DM qd_opt_2 = res_qpsolver2_["x"](ca::Slice(0, n_q_));
        MatrixXd qd_opt_hat_2_tmp = casDM_to_eig(qd_opt_2);

        qd_opt = qd_opt_hat_2_tmp;
      }

      auto start_update_time = std::chrono::high_resolution_clock::now();

      // * 1.0)));
      q_it = pin::integrate(mdl_, q_it, VectorXd(qd_opt * dt_));

      // Computing updated errors
      // FK
      pin::framesForwardKinematics(mdl_, mdl_data_, q_it);

      if (constrained_control_)
      {
        // Computing RCM error
        B_x_Fprercm = mdl_data_.oMf[id_Fprercm_];
        B_x_Fpostrcm = mdl_data_.oMf[id_Fpostrcm_];

        pin::computeFrameJacobian(mdl_, mdl_data_, q_it, id_Fprercm_,
                                  pin::ReferenceFrame::LOCAL_WORLD_ALIGNED,
                                  B_Jb_Fprercm);
        pin::computeFrameJacobian(mdl_, mdl_data_, q_it, id_Fpostrcm_,
                                  pin::ReferenceFrame::LOCAL_WORLD_ALIGNED,
                                  B_Jb_Fpostrcm);

        computeRCMTaskError(B_x_Fprercm, B_x_Fpostrcm, B_Jb_Fprercm,
                            B_Jb_Fpostrcm, err_rcm, B_Jb_Frcm);
        if (verb_level_ >= 1)
          std::cout << "Error RCM: " << err_rcm << std::endl;
      }

      // Computing EE error
      B_x_Fee = mdl_data_.oMf[id_Fee_];

      computeEETaskError(B_x_Fee, x_d, err_ee);

      if (verb_level_ >= 1)
        std::cout << "[HQP] EE error: " << err_ee.norm() << " -> "
                  << err_ee.transpose() << std::endl;

      B_Jb_Fee.setZero();
      pin::computeFrameJacobian(mdl_, mdl_data_, q_it, id_Fee_,
                                pin::ReferenceFrame::WORLD, B_Jb_Fee);
      double mp = sqrt((B_Jb_Fee * B_Jb_Fee.transpose()).determinant());

      // Check for convergence
      if (constrained_control_)
      {
        if (err_rcm < rcm_error_max_ && err_ee.norm() < max_error_)
        {
          if (verb_level_ >= 1)
          {
            ROS_INFO_STREAM("[HQP] Iteration: "
                            << it << " RCM error: " << err_rcm << " EE error: "
                            << err_ee.norm() << " Manipulability: " << mp);
            ROS_INFO_STREAM("[HQP] Iteration: " << it << " - Solution found "
                                                << q_it);
          }
          success_ = true;
          computeEETaskError(B_x_Fee, x_d, "log3", err_log3_ee);
          break;
        }
      }
      else if (err_ee.norm() < max_error_)
      {
        if (verb_level_ >= 1)
        {
          ROS_INFO_STREAM(
              "[HQP] Iteration: " << it << " EE error: " << err_ee.norm());
          ROS_INFO_STREAM("[HQP] iteration: " << it << " - Solution found "
                                              << q_it.transpose());
        }
        success_ = true;
        computeEETaskError(B_x_Fee, x_d, "log3", err_log3_ee);
        break;
      }

      diff = std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::high_resolution_clock::now() - start_solve_time);
      time_left = max_time_ - diff.count() / 1000000.0;

      if (time_left < 0)
      {
        if (verb_level_ >= 1)
        {
          std::cout << "[HQP] Aborted. Maximum time exceeded: " << time_left
                    << std::endl;
          std::cout << "[HQP] Iteration: " << it
                    << " EE error: " << err_ee.norm() << std::endl;
        }
        break;
      }

      if (aborted_)
      {
        if (verb_level_ >= 1)
          std::cout << "[HQP] Aborted by other IK solver " << std::endl;
        break;
      }

      if (it >= max_iter_)
      {
        if (verb_level_ >= 1)
          std::cout << "[HQP] Aborted. Maximum number of iteration reached. "
                    << std::endl;
        break;
      }

      start_update_time = std::chrono::high_resolution_clock::now();

      // JointVector grad_mp;
      grad_mp.setZero();
      double mp_dq = 1e-3;
      VectorXd Ei;
      Ei = VectorXd::Zero(n_q_);
      for (int joint = 0; joint < grad_mp.size(); joint++)
      {
        Ei.setZero();
        Ei[joint] = 1;

        pin::Data::Matrix6x Jb_mdelta(6, mdl_.nv);
        Jb_mdelta.setZero();
        pin::computeFrameJacobian(mdl_, mdl_data_, q_it - mp_dq * Ei, id_Fee_,
                                  pin::ReferenceFrame::WORLD, Jb_mdelta);
        double mp_mdelta =
            sqrt((Jb_mdelta * Jb_mdelta.transpose()).determinant());

        pin::Data::Matrix6x Jb_pdelta(6, mdl_.nv);
        Jb_pdelta.setZero();
        pin::computeFrameJacobian(mdl_, mdl_data_, q_it + mp_dq * Ei, id_Fee_,
                                  pin::ReferenceFrame::WORLD, Jb_pdelta);
        double mp_pdelta =
            sqrt((Jb_pdelta * Jb_pdelta.transpose()).determinant());

        grad_mp[joint] = (mp_pdelta - mp_mdelta) / (2 * mp_dq);
      }
    }

    if (success_)
    {
      diff = std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::high_resolution_clock::now() - start_solve_time);

      if (verb_level_ >= 1)
        std::cout << "HQP: Convergence achieved!" << std::endl;
      q_sol = q_it;
      if (logging_enabled_)
      {
        fout_ << prob_id_ << "," << diff.count() << "," << err_rcm << ","
              << err_ee.norm() << "," << err_log3_ee.head(3).norm() << ","
              << err_log3_ee.tail(3).norm() << "," << B_x_Fee.translation()[0]
              << "," << B_x_Fee.translation()[1] << ","
              << B_x_Fee.translation()[2] << "," << B_x_Fee.rotation()(0, 0)
              << "," << B_x_Fee.rotation()(1, 0) << ","
              << B_x_Fee.rotation()(2, 0) << "," << B_x_Fee.rotation()(0, 1)
              << "," << B_x_Fee.rotation()(1, 1) << ","
              << B_x_Fee.rotation()(2, 1) << "\n";
      }
      return 0;
    }

    if (verb_level_ >= 1)
      std::cout << "\nHQP: Warning: the iterative algorithm has not reached "
                   "convergence to the desired precision "
                << std::endl;
    return 1;
  }

  void HQP_CASADI::generate_qp_solver()
  {
    // Optimizer options
    ca::Dict opts;
    opts["osqp.max_iter"] = 1000;
    opts["error_on_fail"] = false; // true
    if (verb_level_ >= 1)
    {
      opts["verbose"] = true;      // false
      opts["osqp.verbose"] = true; //
    }
    else
    {
      opts["verbose"] = false;      // false
      opts["osqp.verbose"] = false; //
    }
    opts["warm_start_primal"] = hqp_warm_start_; // 0

    // Optimization Variable
    ca::SX x1 = ca::SX::sym("x", 3 * n_q_, 1);

    // Fixed parameters
    ca::SX par1 = ca::SX::sym("par", 3 * n_q_, 5 * n_q_ + 2);

    ca::SX Q1 = par1(ca::Slice(), ca::Slice(0, 3 * n_q_));
    ca::SX p1 = par1(ca::Slice(), 3 * n_q_);
    ca::SX C1 = (par1(ca::Slice(), ca::Slice(3 * n_q_ + 1, 5 * n_q_ + 1))).T();
    ca::SX d1 = par1(ca::Slice(0, 2 * n_q_), par1.size2() - 1);

    // Nonlinear problem declaration
    ca::SXDict qp1;

    qp1 = {
        {"x", x1},
        {"f", 0.5 * ca::SX::mtimes(x1.T(), ca::SX::mtimes(Q1, x1)) +
                  ca::SX::mtimes(hqp_Ke1_, ca::SX::mtimes(p1.T(), x1)) +
                  ca::SX::mtimes(hqp_Kd1_, ca::SX::mtimes(x1.T(), x1))},
        {"g", ca::SX(d1) - ca::SX::mtimes(C1, x1)},
        {"p", par1},
    };

    qpsolver1_ = qpsol("qpsol1", "osqp", qp1, opts);

    // Optimization Variable
    ca::SX x2 = ca::SX::sym("x", 3 * n_q_, 1);

    // Fixed parameters
    ca::SX par2 = ca::SX::sym("par", 3 * n_q_, 5 * n_q_ + 2);

    ca::SX Q2 = par2(ca::Slice(), ca::Slice(0, 3 * n_q_));
    ca::SX p2 = par2(ca::Slice(), 3 * n_q_);
    ca::SX C2 = (par2(ca::Slice(), ca::Slice(3 * n_q_ + 1, 5 * n_q_ + 1))).T();
    ca::SX d2 = par2(ca::Slice(0, 2 * n_q_), par2.size2() - 1);

    // Nonlinear problem declaration
    ca::SXDict qp2;

    // Nonlinear problem arguments definition
    qp2 = {{"x", x2},
           {"f", 0.5 * ca::SX::mtimes(x2.T(), ca::SX::mtimes(Q2, x2)) +
                     ca::SX::mtimes(hqp_Ke2_, ca::SX::mtimes(p2.T(), x2)) +
                     ca::SX::mtimes(hqp_Kd2_, ca::SX::mtimes(x2.T(), x2))},
           {"g", ca::SX(d2) - ca::SX::mtimes(C2, x2)},
           {"p", par2}};

    qpsolver2_ = qpsol("qpsol2", "osqp", qp2, opts);
  }

  //* Casadi-Eigen conversion functions

  ca::DM HQP_CASADI::eig_to_casDM(const VectorXd &eig)
  {
    auto dm = casadi::DM(casadi::Sparsity::dense(eig.size()));
    for (int i = 0; i < eig.size(); i++)
    {
      dm(i) = eig(i);
    }
    return dm;
  }

  ca::DM HQP_CASADI::eigmat_to_casDM(const MatrixXd &eig)
  {
    casadi::DM dm = casadi::DM(casadi::Sparsity::dense(eig.rows(), eig.cols()));
    std::copy(eig.data(), eig.data() + eig.size(), dm.ptr());
    return dm;
  }

  MatrixXd HQP_CASADI::casDM_to_eig(const casadi::DM &dm)
  {
    auto vector_x = static_cast<std::vector<double>>(dm);
    MatrixXd eig = MatrixXd::Zero(dm.size1(), dm.size2());

    for (int i = 0; i < dm.size1(); i++)
    {
      for (int j = 0; j < dm.size2(); j++)
      {
        eig(i, j) = vector_x[i + j * dm.size2()];
      }
    }
    return eig;
  }

  double HQP_CASADI::manipEE(pin::Model model, pin::Data data, VectorXd q,
                             pin::FrameIndex ee_id)
  {
    pin::Data::Matrix6x Jb(6, model.nv);
    pin::computeFrameJacobian(model, data, q, ee_id, pin::ReferenceFrame::WORLD,
                              Jb);
    return sqrt((Jb * Jb.transpose()).determinant());
  }

  VectorXd HQP_CASADI::gradManipEE(pin::Model model, pin::Data data, VectorXd q,
                                   pin::FrameIndex ee_id)
  {

    VectorXd grad_mp = VectorXd::Zero(n_q_);
    double mp_dq = 1e-3;
    for (int joint = 0; joint < grad_mp.size(); joint++)
    {
      VectorXd Ei = VectorXd::Zero(n_q_);
      Ei.setZero();
      Ei[joint] = 1;
      grad_mp[joint] = (manipEE(model, data, q + mp_dq * Ei, ee_id) -
                        manipEE(model, data, q - mp_dq * Ei, ee_id)) /
                       (2 * mp_dq);
    }

    return grad_mp;
  }

  // Taken from https://armarx.humanoids.kit.edu/pinv_8hh_source.html
  MatrixXd HQP_CASADI::pseudoInverse(const Eigen::MatrixXd &a, double epsilon)
  {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(a, Eigen::ComputeThinU |
                                                 Eigen::ComputeThinV);

    double tolerance = epsilon * std::max(a.cols(), a.rows()) *
                       svd.singularValues().array().abs()(0);

    return svd.matrixV() *
           (svd.singularValues().array().abs() > tolerance)
               .select(svd.singularValues().array().inverse(), 0)
               .matrix()
               .asDiagonal() *
           svd.matrixU().adjoint();
  }

} // namespace coiks

#endif
