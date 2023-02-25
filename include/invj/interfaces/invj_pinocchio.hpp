/******************************************************************************
# invj_pinocchio.hpp:  CoIKS Inverse Jacobian IK solver                       #
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

#ifndef INVJ_PINOCCHIO_HPP
#define INVJ_PINOCCHIO_HPP

// Source HPP
#include <coiks/solver_base.hpp>
#include <coiks/solver_options.hpp>
// Pinocchio
#include <pinocchio/algorithm/joint-configuration.hpp>
// C++
#include <chrono>

namespace coiks
{

  class COIKS;

  class INVJ_PINOCCHIO : public SolverBase
  {
    friend class coiks::COIKS;

  public:
    INVJ_PINOCCHIO(const pin::Model &_model, const pin::FrameIndex &_Fid,
                   SolverOptions _solver_opts, const double _max_time,
                   const double _max_error, const int _max_iter = 1000,
                   const double _dt = 1);
    ~INVJ_PINOCCHIO();

    int IkSolve(const VectorXd &q_init, const pin::SE3 &x_des, VectorXd &q_sol);
    MatrixXd pseudoInverse(const Eigen::MatrixXd &a, double epsilon);
    MatrixXd weightedPseudoInverse(const Eigen::MatrixXd &a,
                                   const Eigen::VectorXd w);
    void GetOptions(SolverOptions _solver_opts);

    int computeEETaskError(const pin::SE3 &B_x_Fee, const pin::SE3 &x_d,
                           Vector6d &err_ee);
    int computeEETaskError(const pin::SE3 &B_x_Fee, const pin::SE3 &x_d,
                           std::string err_type, Vector6d &err_ee);
    int computeRCMTaskError(const pin::SE3 &B_x_Fprercm,
                            const pin::SE3 &B_x_Fpostrcm,
                            const MatrixXd &B_Jb_Fprercm,
                            const MatrixXd &B_Jb_Fpostrcm, double &err_rcm,
                            VectorXd &B_Jb_Frcm);

    inline void abort() { aborted_ = true; }
    inline void reset() { aborted_ = false; }
    inline void set_max_time(double _max_time) { max_time_ = _max_time; };
    inline void set_max_error(double _max_error) { max_error_ = _max_error; };

  private:
    pin::Model mdl_;
    pin::Data mdl_data_;
    pin::FrameIndex id_Fee_;
    pin::FrameIndex id_Fprercm_;
    pin::FrameIndex id_Fpostrcm_;

    int n_q_;
    int max_iter_;
    int verb_level_;

    double max_time_;
    double max_error_;
    double dt_;

    bool aborted_;
    bool success_;
    bool constrained_control_;
    bool logging_enabled_;

    std::string log_path_;

    VectorXd q_ul_;
    VectorXd q_ll_;
    VectorXd q_sol_;

    std::string error_type_;
    std::string prercm_joint_name_;
    std::string postrcm_joint_name_;

    double rcm_error_max_;
    double Kee_;
    double Krcm_;

    // Jacobians
    MatrixXd Jb_task_1_; // RCM
    MatrixXd Jb_task_2_; // EE Tracking
    MatrixXd pinv_Jtask_1_;
    MatrixXd pinv_Jtask_2_;

    Vector3d B_p_Ftrocar_;

    // logger output
    std::fstream fout_;
    int prob_id_;
  };

  void INVJ_PINOCCHIO::GetOptions(SolverOptions _solver_opts)
  {
    SolverOptions so;

    so = _solver_opts;

    // Solver parameters
    error_type_ = so.error_type_;

    // Constraint control
    constrained_control_ = so.constrained_control_;
    prercm_joint_name_ = so.prercm_joint_name_;
    postrcm_joint_name_ = so.postrcm_joint_name_;
    B_p_Ftrocar_ = so.trocar_pos_;
    rcm_error_max_ = so.rcm_error_max_;

    // INVJ variables
    Krcm_ = so.invj_Ke1_;
    Kee_ = so.invj_Ke2_;

    // Logging
    verb_level_ = so.verb_level_;
    logging_enabled_ = so.logging_enabled_;
    log_path_ = so.log_path_;

    std::cout << "\n------\nINVJ Options summary:" << std::endl;
    std::cout << "Constrained control enabled: " << constrained_control_
              << std::endl;
    std::cout << "Max. RCM error: " << rcm_error_max_ << std::endl;
    std::cout << "Max. EE error: " << max_error_ << std::endl;
    std::cout << "Coeff. Krcm (RCM error) : " << Krcm_ << std::endl;
    std::cout << "Coeff. Kee (EE error): " << Kee_ << std::endl;

    std::cout << "Error computation type: " << error_type_ << std::endl;
    std::cout << "Logging enabled: " << logging_enabled_ << std::endl;
    std::cout << "Verbosity Level: " << verb_level_ << std::endl;

    if (constrained_control_)
    {
      std::cout << "Pre-RCM joint: " << prercm_joint_name_ << std::endl;
      std::cout << "Post-RCM joint: " << postrcm_joint_name_ << std::endl;
      std::cout << "Trocar Position: " << B_p_Ftrocar_.transpose() << std::endl;
      std::cout << "Max. RCM error: " << rcm_error_max_ << std::endl;
    }
  }

  INVJ_PINOCCHIO::INVJ_PINOCCHIO(const pin::Model &_model,
                                 const pin::FrameIndex &_Fee_id,
                                 SolverOptions _solver_opts,
                                 const double _max_time,
                                 const double _max_error, const int _max_iter,
                                 const double _dt)
      : aborted_(false), success_(false), mdl_(_model), n_q_(_model.nq),
        id_Fee_(_Fee_id), max_time_(_max_time), max_error_(_max_error),
        max_iter_(_max_iter), dt_(_dt)
  {
    std::cout << "----------\nInitializing IK solver COIKS-INVJ" << std::endl;

    GetOptions(_solver_opts);

    mdl_data_ = pin::Data(mdl_);
    if (constrained_control_)
    {
      id_Fpostrcm_ = mdl_.getFrameId(postrcm_joint_name_);
      id_Fprercm_ = mdl_.getFrameId(prercm_joint_name_);
    }
    q_ll_ = mdl_.lowerPositionLimit;
    q_ul_ = mdl_.upperPositionLimit;

    prob_id_ = 0;
    Jb_task_1_.resize(1, n_q_);
    pinv_Jtask_1_.resize(n_q_, 1);

    if (error_type_ == "only_p")
    {
      Jb_task_2_.resize(3, n_q_);
      pinv_Jtask_2_.resize(n_q_, 3);
    }
    else
    {
      Jb_task_2_.resize(6, n_q_);
      pinv_Jtask_2_.resize(n_q_, 6);
    }
    if (logging_enabled_)
    {
      time_t currentTime;
      struct tm *localTime;

      time(&currentTime); // Get the current time
      localTime = localtime(&currentTime);

      fout_.open(log_path_ + "log_invj_" + std::to_string(localTime->tm_mday) +
                     std::to_string(localTime->tm_hour) +
                     std::to_string(localTime->tm_min) +
                     std::to_string(localTime->tm_sec) + ".csv",
                 std::ios::out | std::ios::app);
      fout_ << "idx,time,error_rcm,error_ee,error_t,error_r,p_x,p_y,p_z,r_11,r_"
               "21,r_31,r_12,r_22,r_32\n";
    }
  }

  INVJ_PINOCCHIO::~INVJ_PINOCCHIO()
  {
    if (logging_enabled_)
      fout_.close();
  }

  // Taken from https://armarx.humanoids.kit.edu/pinv_8hh_source.html
  MatrixXd INVJ_PINOCCHIO::pseudoInverse(const Eigen::MatrixXd &a,
                                         double epsilon)
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

  // Taken from https://armarx.humanoids.kit.edu/pinv_8hh_source.html
  MatrixXd INVJ_PINOCCHIO::weightedPseudoInverse(const Eigen::MatrixXd &a,
                                                 const Eigen::VectorXd w)
  {
    int lenght = w.size();

    Eigen::DiagonalMatrix<double, Eigen::Dynamic> Winv(lenght);
    Winv = w.asDiagonal().inverse(); // diag(1./w)

    Eigen::MatrixXd tmp(lenght, lenght);

    tmp = pseudoInverse(a * Winv * a.transpose(), 10E-10);

    return Winv * a.transpose() * tmp;
  }

  int INVJ_PINOCCHIO::computeEETaskError(const pin::SE3 &B_x_Fee,
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
      Vector3d err_tr = (x_d.translation() - B_x_Fee.translation());
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

  int INVJ_PINOCCHIO::computeEETaskError(const pin::SE3 &B_x_Fee,
                                         const pin::SE3 &x_d,
                                         std::string err_type, Vector6d &err_ee)
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
      Vector3d err_tr = (x_d.translation() - B_x_Fee.translation());
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

  int INVJ_PINOCCHIO::computeRCMTaskError(const pin::SE3 &B_x_Fprercm,
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

    //* Computation of RCM Error
    Vector3d pe = B_p_Ftrocar_ - B_p_Frcm;
    Vector3d pe_hat = pe.normalized();

    err_rcm = pe.norm();

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

  int INVJ_PINOCCHIO::IkSolve(const VectorXd &q_init, const pin::SE3 &x_d,
                              VectorXd &q_sol)
  {
    std::chrono::time_point<std::chrono::high_resolution_clock>
        start_solve_time = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds diff;

    prob_id_ += 1;

    double time_left = 0.0;

    q_sol = q_init;
    VectorXd q_it = q_init;

    if (verb_level_ >= 1)
      std::cout << "\n[INVJ] Solving with optimizer INVJ IK and q_init:"
                << q_init.transpose() << std::endl;

    Vector6d err_ee;
    Vector6d err_log3_ee;
    double err_rcm = 0.0;
    VectorXd q_dot(mdl_.nv);

    pin::Data::Matrix6x B_Jb_Fee(6, n_q_);
    pin::Data::Matrix6x B_Jb_Fprercm(6, n_q_);
    pin::Data::Matrix6x B_Jb_Fpostrcm(6, n_q_);
    VectorXd B_Jb_Frcm(1, n_q_);

    B_Jb_Fee.setZero();
    B_Jb_Fprercm.setZero();
    B_Jb_Fpostrcm.setZero();
    B_Jb_Frcm.setZero();

    pin::SE3 B_x_Fee;
    pin::SE3 B_x_Fprercm;
    pin::SE3 B_x_Fpostrcm;

    success_ = false;

    Jb_task_1_.setZero();
    Jb_task_2_.setZero();

    //* Updating FK
    pin::framesForwardKinematics(mdl_, mdl_data_, q_it);
    B_x_Fee = mdl_data_.oMf[id_Fee_];

    if (constrained_control_)
    {
      B_x_Fprercm = mdl_data_.oMf[id_Fprercm_];
      B_x_Fpostrcm = mdl_data_.oMf[id_Fpostrcm_];
    }

    diff = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start_solve_time);
    time_left = max_time_ - diff.count() / 1000000.0;

    for (int it = 0;; it++)
    {
      //* Computing FK
      pin::framesForwardKinematics(mdl_, mdl_data_, q_it);
      B_x_Fee = mdl_data_.oMf[id_Fee_];

      //* Computing EE error
      computeEETaskError(B_x_Fee, x_d, err_ee);
      if (verb_level_ >= 1)
        std::cout << "[INVJ] [" << it << "] error EE: " << err_ee.norm()
                  << " : " << err_ee.transpose() << std::endl;

      //* Verifying convergence
      if (constrained_control_)
      {
        if (err_ee.norm() < max_error_ && err_rcm < rcm_error_max_)
        {
          if (verb_level_ >= 1)
            std::cout << "[INVJ]  Solution found" << std::endl;
          success_ = true;
          computeEETaskError(B_x_Fee, x_d, "log3", err_log3_ee);
          break;
        }
      }
      else
      {
        if (err_ee.norm() < max_error_)
        {
          if (verb_level_ >= 1)
            std::cout << "[INVJ]  Solution found" << std::endl;
          success_ = true;
          computeEETaskError(B_x_Fee, x_d, "log3", err_log3_ee);
          break;
        }
      }

      // * Checking exitting conditions
      if (time_left < 0)
      {
        if (verb_level_ >= 1)
          std::cout << "[INVJ]  Aborted. Maximum time exceeded: " << time_left
                    << std::endl;
        break;
      }

      if (aborted_)
      {
        if (verb_level_ >= 1)
          std::cout << "[INVJ]  Aborted by other IK solver " << std::endl;
        break;
      }

      if (it >= max_iter_)
      {
        if (verb_level_ >= 1)
          std::cout << "[INVJ]  Aborted. Maximum number of iteration reached. "
                    << std::endl;
        break;
      }

      //*  Computing Jacobians
      pin::computeFrameJacobian(mdl_, mdl_data_, q_it, id_Fee_,
                                pin::ReferenceFrame::WORLD, B_Jb_Fee);

      //* Defining Task Jacobians
      //? Tracking in 3D?
      if (error_type_ == "only_p")
        Jb_task_2_ = B_Jb_Fee.topRows(3);
      else
        Jb_task_2_ = B_Jb_Fee;

      //* Computing pseudoinverses
      pinv_Jtask_2_ = pseudoInverse(Jb_task_2_, 1e-10);

      //? For constrained control
      if (constrained_control_)
      {
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
          std::cout << "[INVJ] [" << it << "] error RCM : " << err_rcm
                    << std::endl;

        //* Defining Task Jacobians
        Jb_task_1_ = B_Jb_Frcm.transpose();

        //* Computing pseudoinverses
        pinv_Jtask_1_ = pseudoInverse(Jb_task_1_, 1e-10);
      }

      //? Use constrained control?
      if (constrained_control_)
      {
        //? Tracking in 3D?
        if (error_type_ == "only_p")
        {
          q_dot.noalias() =
              pinv_Jtask_1_ * Krcm_ * err_rcm +
              (MatrixXd::Identity(n_q_, n_q_) - pinv_Jtask_1_ * Jb_task_1_) *
                  pinv_Jtask_2_ * Kee_ * err_ee.head(3);
        }
        else
        {
          q_dot.noalias() =
              pinv_Jtask_1_ * Krcm_ * err_rcm +
              (MatrixXd::Identity(n_q_, n_q_) - pinv_Jtask_1_ * Jb_task_1_) *
                  pinv_Jtask_2_ * Kee_ * err_ee;
        }
      }
      else
      {
        //? Use non-constrained control
        q_dot.noalias() = pinv_Jtask_2_ * Kee_ * err_ee;
      }

      VectorXd q_prev = q_it;
      q_it = pin::integrate(mdl_, q_prev, q_dot * dt_);

      for (int i = 0; i < n_q_; i++)
      {
        if (q_it[i] > q_ul_[i])
        {
          q_it = pin::randomConfiguration(mdl_);
        }

        if (q_it[i] < q_ll_[i])
        {
          q_it = pin::randomConfiguration(mdl_);
        }
      }

      diff = std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::high_resolution_clock::now() - start_solve_time);
      time_left = max_time_ - diff.count() / 1000000.0;
    }

    if (success_)
    {
      diff = std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::high_resolution_clock::now() - start_solve_time);
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
    else
    {
      if (verb_level_ >= 1)
        std::cout << "\[INVJ] Warning: the iterative algorithm has not reached "
                     "convergence to the desired precision "
                  << std::endl;
      return 1;
    }
  }
} // namespace coiks

#endif