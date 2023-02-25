/******************************************************************************
# nlo_casadi.hpp.hpp:  CoIKS Nonlinear optimization IK solver                 #
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

#ifndef NLO_CASADI_HPP
#define NLO_CASADI_HPP

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

  class NLO_CASADI : public SolverBase
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
    const int n_q_;

    int verb_level_;
    std::string time_stats_;

    VectorXd q_ul_;
    VectorXd q_ll_;
    VectorXd q_sol_;
    std::vector<double> q_sol_vec_;
    VectorXd q_max_;
    VectorXd q_min_;

    bool aborted_;
    bool success_;
    bool constrained_control_;
    bool rcm_is_cost_;
    bool logging_enabled_;
    bool nlo_concurrent_;
    bool manipulability_enabled_;

    std::string log_path_;

    Vector3d B_p_Ftrocar_;

    std::string nlo_linear_solver_;
    std::string error_type_;
    std::string prercm_joint_name_;
    std::string postrcm_joint_name_;

    // Penalty gains
    double mu0_;
    double mu1_;
    double mu2_;
    double mu3_;
    double mu4_;

    int nlo_concurrent_iterations_;
    std::string nlo_warm_start_;

    ca::Function ca_perr2_;
    ca::Function ca_elog3_ee_;
    ca::Function ca_ercm_;
    ca::Function solver_;
    ca::Function ca_log3_;
    ca::Function ca_log6_;
    ca::Function FK_;
    ca::Function JB_Fee_;
    ca::Function FK_Fpre_;
    ca::Function FK_Fpost_;

    double rcm_error_max_;
    std::vector<double> cost_coeff_;
    // logger output
    std::fstream fout_;
    int prob_id_;

  public:
    NLO_CASADI(const pin::Model &_model, const pin::FrameIndex &_Fee_id,
               SolverOptions _solver_opts, const double _max_time,
               const double _max_error, const int _max_iter, const double _dt);
    ~NLO_CASADI();

    int IkSolve(const VectorXd q_init, const pin::SE3 &x_d, VectorXd &q_sol);

    ca::DM eig_to_casDM(const VectorXd &eig);
    ca::DM eigmat_to_casDM(const MatrixXd &eig);
    MatrixXd casDM_to_eig(const casadi::DM &cas);

    void generate_ca_RCM_error();
    void generate_ca_EE_error();
    void generate_ca_log3_EE_error();
    void generate_ca_log3();
    void generate_ca_log6();
    void generate_nlsolver();
    void GetOptions(SolverOptions _solver_opts);

    inline void abort()
    {
      aborted_ = true;
      if (verb_level_ >= 1)
        std::cout << "Setting NLO abort" << std::endl;
    }
    inline void reset() { aborted_ = false; }
    inline void set_max_time(double _max_time) { max_time_ = _max_time; }
    inline void set_max_error(double _max_error) { max_error_ = _max_error; }
  };

  void NLO_CASADI::GetOptions(SolverOptions _solver_opts)
  {
    SolverOptions so;

    so = _solver_opts;
    cost_coeff_.clear();

    // Solver parameters
    manipulability_enabled_ = so.manipulability_enabled_;
    nlo_linear_solver_ = so.nlo_linear_solver_;
    nlo_concurrent_ = so.nlo_concurrent_;
    nlo_concurrent_iterations_ = so.nlo_concurrent_iterations_;
    nlo_warm_start_ = so.nlo_warm_start_;

    // error_type_ = "log3";
    error_type_ = so.nlo_error_type_;
    cost_coeff_ = so.cost_coeff_;

    // Constrained Control
    constrained_control_ = so.constrained_control_;
    B_p_Ftrocar_ = so.trocar_pos_;
    rcm_error_max_ = so.rcm_error_max_;
    rcm_is_cost_ = true;
    prercm_joint_name_ = so.prercm_joint_name_;
    postrcm_joint_name_ = so.postrcm_joint_name_;

    // Logging
    time_stats_ = so.time_stats_;
    verb_level_ = so.verb_level_;
    logging_enabled_ = so.logging_enabled_;
    log_path_ = so.log_path_;
    std::cout << "\n------\n NLO Options summary:" << std::endl;
    std::cout << "Constrained control enabled: " << constrained_control_
              << std::endl;
    std::cout << "NLO concurrent mode enabled: " << nlo_concurrent_
              << std::endl;
    std::cout << "NLO concurrent iterations: " << nlo_concurrent_iterations_
              << std::endl;
    std::cout << "NLO warm start: " << nlo_warm_start_ << std::endl;
    std::cout << "Manipulability enabled: " << manipulability_enabled_
              << std::endl;
    std::cout << "Error computation type: " << error_type_ << std::endl;
    std::cout << "Linear solver: " << nlo_linear_solver_ << std::endl;
    std::cout << "Max. EE error: " << max_error_ << std::endl;
    std::cout << "Verbosity Level: " << verb_level_ << std::endl;
    std::cout << "Time Statstics: " << time_stats_ << std::endl;
    std::cout << "Logging enabled: " << logging_enabled_ << std::endl;
    std::cout << "Coeff[0] Pos(log3)/Pose(log6) error: " << cost_coeff_[0]
              << std::endl;
    std::cout << "Coeff[1] Ori(log3): " << cost_coeff_[1] << std::endl;
    std::cout << "Coeff[2] RCM error: " << cost_coeff_[2] << std::endl;
    std::cout << "Coeff[3] Qdelta: " << cost_coeff_[3] << std::endl;
    std::cout << "Coeff[4] Manipulability: " << cost_coeff_[4] << std::endl;
    if (constrained_control_)
    {
      std::cout << "Max. RCM error: " << rcm_error_max_ << std::endl;
      std::cout << "Pre-RCM joint: " << prercm_joint_name_ << std::endl;
      std::cout << "Post-RCM joint: " << postrcm_joint_name_ << std::endl;
      std::cout << "Trocar Position: " << B_p_Ftrocar_.transpose() << std::endl;
    }
  }

  NLO_CASADI::NLO_CASADI(const pin::Model &_model,
                         const pin::FrameIndex &_Fee_id,
                         SolverOptions _solver_opts, const double _max_time,
                         const double _max_error, const int _max_iter,
                         const double _dt)
      : aborted_(false), success_(false), mdl_(_model), n_q_(_model.nq),
        id_Fee_(_Fee_id), max_time_(_max_time), max_error_(_max_error),
        max_iter_(_max_iter), dt_(_dt)
  {
    std::cout << "----------\nInitializing IK solver COIKS-NLO" << std::endl;

    GetOptions(_solver_opts);

    mu0_ = cost_coeff_[0]; // 10
    mu1_ = cost_coeff_[1]; // 0.005
    mu2_ = cost_coeff_[2]; // 0.001
    mu3_ = cost_coeff_[3]; // 100
    mu4_ = cost_coeff_[4]; // 100

    mdl_data_ = pin::Data(mdl_);

    q_max_ = mdl_.upperPositionLimit;
    q_min_ = mdl_.lowerPositionLimit;

    if (constrained_control_)
    {
      id_Fpostrcm_ = mdl_.getFrameId(postrcm_joint_name_);
      id_Fprercm_ = mdl_.getFrameId(prercm_joint_name_);
    }

    //* Exporting Kinematics Casadi Functions
    //? Generate Functions
    // Cast the model into casadi::SX
    pin::ModelTpl<ca::SX> model = mdl_.cast<ca::SX>();
    // Create Data model as casadi::SX
    pinocchio::DataTpl<ca::SX> data(model);
    // Create casadi::SX joint variable
    ca::SX ca_q = ca::SX::sym("ca_q", n_q_, 1);
    // Create associated Eigen matrix
    Eigen::Matrix<ca::SX, Eigen::Dynamic, 1> _q;
    _q.resize(n_q_, 1);
    // Copy casadi::SX into Eigen::Matrix
    pin::casadi::copy(ca_q, _q);

    //* Generate symbolic FK
    std::cout << "Generate Casadi FK function" << std::endl;
    pin::framesForwardKinematics(model, data, _q);
    // Extract Eigen::Matrix results
    Eigen::Matrix<ca::SX, 3, 1> eig_fk_pos = data.oMf.at(id_Fee_).translation();
    Eigen::Matrix<ca::SX, 3, 3> eig_fk_rot = data.oMf.at(id_Fee_).rotation();
    // Create associated casadi::SX variables

    ca::SX ca_fk_tr =
        ca::SX(ca::Sparsity::dense(eig_fk_pos.rows(), eig_fk_pos.cols()));
    ca::SX ca_fk_rot =
        ca::SX(ca::Sparsity::dense(eig_fk_rot.rows(), eig_fk_rot.cols()));
    // Copy Eigen::Matrix into casadi::SX
    pinocchio::casadi::copy(eig_fk_pos, ca_fk_tr);
    pinocchio::casadi::copy(eig_fk_rot, ca_fk_rot);
    // Generate function
    FK_ = ca::Function("forward_kinematics", {ca_q}, {ca_fk_tr, ca_fk_rot},
                       {"q"}, {"ee_pos", "ee_rot"});

    generate_ca_log3();
    generate_ca_log6();
    generate_ca_EE_error();
    generate_ca_log3_EE_error();

    if (constrained_control_)
    {

      //* Generate function
      std::cout << "Generate Casadi FK prercm" << std::endl;
      // Extract Eigen::Matrix results
      Eigen::Matrix<ca::SX, 3, 1> eig_fk_p_prercm =
          data.oMf.at(id_Fprercm_).translation();
      Eigen::Matrix<ca::SX, 3, 3> eig_fk_R_prercm =
          data.oMf.at(id_Fprercm_).rotation();
      // Create associated casadi::SX variables
      ca::SX ca_FK_p_Fpre = ca::SX(
          ca::Sparsity::dense(eig_fk_p_prercm.rows(), eig_fk_p_prercm.cols()));
      ca::SX ca_FK_R_Fpre = ca::SX(
          ca::Sparsity::dense(eig_fk_R_prercm.rows(), eig_fk_R_prercm.cols()));
      // Copy Eigen::Matrix into casadi::SX
      pinocchio::casadi::copy(eig_fk_p_prercm, ca_FK_p_Fpre);
      pinocchio::casadi::copy(eig_fk_R_prercm, ca_FK_R_Fpre);

      FK_Fpre_ = ca::Function("forward_kinematics_prercm", {ca_q},
                              {ca_FK_p_Fpre, ca_FK_R_Fpre}, {"q"},
                              {"p_prercm", "R_prercm"});

      //* Generate function
      std::cout << "Generate Casadi FK postrcm" << std::endl;
      // Extract Eigen::Matrix results
      Eigen::Matrix<ca::SX, 3, 1> eig_fk_p_postrcm =
          data.oMf.at(id_Fpostrcm_).translation();
      Eigen::Matrix<ca::SX, 3, 3> eig_fk_R_postrcm =
          data.oMf.at(id_Fpostrcm_).rotation();
      // Create associated casadi::SX variables
      ca::SX ca_FK_p_Fpost = ca::SX(ca::Sparsity::dense(
          eig_fk_p_postrcm.rows(), eig_fk_p_postrcm.cols()));
      ca::SX ca_FK_R_Fpost = ca::SX(ca::Sparsity::dense(
          eig_fk_R_postrcm.rows(), eig_fk_R_postrcm.cols()));
      // Copy Eigen::Matrix into casadi::SX
      pinocchio::casadi::copy(eig_fk_p_postrcm, ca_FK_p_Fpost);
      pinocchio::casadi::copy(eig_fk_R_postrcm, ca_FK_R_Fpost);
      // Generate function
      FK_Fpost_ = ca::Function("forward_kinematics_postrcm", {ca_q},
                               {ca_FK_p_Fpost, ca_FK_R_Fpost}, {"q"},
                               {"p_postrcm", "R_postrcm"});
      generate_ca_RCM_error();
    }
    generate_nlsolver();

    // Getting Joints
    std::cout << "Joints considered for IK solver: " << n_q_ << std::endl;
    std::cout << "Joints lower limits: " << mdl_.lowerPositionLimit.transpose()
              << std::endl;
    std::cout << "Joints upper limits: " << mdl_.upperPositionLimit.transpose()
              << std::endl;

    prob_id_ = 0;

    if (logging_enabled_)
    {
      time_t currentTime;
      struct tm *localTime;

      time(&currentTime); // Get the current time
      localTime = localtime(&currentTime);

      fout_.open(log_path_ + "log_nlo_" + std::to_string(localTime->tm_mday) +
                     std::to_string(localTime->tm_hour) +
                     std::to_string(localTime->tm_min) +
                     std::to_string(localTime->tm_sec) + ".csv",
                 std::ios::out | std::ios::app);
      fout_ << "idx,time,error_rcm,error_ee,error_t,error_r,p_x,p_y,p_z,r_11,r_"
               "21,r_31,r_12,r_22,r_32\n";
    }

    std::cout << "COIKS-NLO initialized\n---------------------------------"
              << std::endl;
  }

  NLO_CASADI::~NLO_CASADI()
  {
    if (logging_enabled_)
      fout_.close();
  }

  int NLO_CASADI::IkSolve(const VectorXd q_init, const pin::SE3 &x_d,
                          VectorXd &q_sol)
  {

    std::chrono::time_point<std::chrono::high_resolution_clock>
        start_solve_time = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds diff;

    prob_id_ += 1;
    double time_left = 0.0;
    success_ = false;
    double err_ee = 0.0;
    double err_log3_p = 0.0;
    double err_log3_R = 0.0;

    ca::DM q_opt;
    ca::DM f_opt;

    q_sol = q_init;
    if (verb_level_ >= 1)
      std::cout << "\nSolving NLO IK with optimizer with q_init:"
                << q_init.transpose() << std::endl;

    ca::SX ca_ps = eig_to_casDM(x_d.translation());
    ca::SX ca_Rd = eigmat_to_casDM(x_d.rotation());
    ca::SX ca_qin = eigmat_to_casDM(q_init);
    ca::SX ca_trocar = eig_to_casDM(B_p_Ftrocar_);

    ca::DMDict arg_nlopt;

    ca::SX ca_qmin = eig_to_casDM(q_min_ - q_init);
    ca::SX ca_qmax = eig_to_casDM(q_max_ - q_init);

    auto x_act = FK_(ca_qin);
    ca::SX p_act = x_act[0];
    ca::SX R_act = x_act[1];

    double err_rcm = 0.0;
    double manip = 0.0;
    VectorXd grad_manip = VectorXd::Zero(mdl_.nv);

    ca::DM par = ca::DM::zeros(2 * n_q_ + 22);
    ca::DM mu({mu0_, mu1_, mu2_, mu3_, mu4_});

    double dT = 0.01;
    par(ca::Slice(0, n_q_)) = ca_qin;
    par(ca::Slice(n_q_, n_q_ + 3)) = ca_ps;
    par(ca::Slice(n_q_ + 3, n_q_ + 12)) = ca::DM::reshape(ca_Rd, 9, 1);
    par(ca::Slice(n_q_ + 12, n_q_ + 15)) = ca_trocar;
    par(ca::Slice(n_q_ + 15, n_q_ + 20)) = mu;
    par(n_q_ + 20) = dT;
    par(n_q_ + 21) = ca::DM(manip);
    par(ca::Slice(n_q_ + 22, 2 * n_q_ + 22)) = eig_to_casDM(grad_manip);

    if (manipulability_enabled_)
    {
      // Computing manipulability
      pin::Data::Matrix6x B_Jb_Fee(6, n_q_);
      B_Jb_Fee.setZero();

      pin::computeFrameJacobian(mdl_, mdl_data_, q_init, id_Fee_,
                                pin::ReferenceFrame::WORLD, B_Jb_Fee);

      manip = sqrt((B_Jb_Fee * B_Jb_Fee.transpose()).determinant());
      if (verb_level_ >= 1)
        std::cout << "Manipulability: " << manip << std::endl;
      // Copmuting manipulability gradient
      grad_manip.setZero();
      double mp_dq = 1e-3;
      VectorXd Ei = VectorXd::Zero(mdl_.nv);

      for (int joint = 0; joint < grad_manip.size(); joint++)
      {
        Ei.setZero();
        Ei[joint] = 1;

        pin::Data::Matrix6x Jb_mdelta(6, n_q_);
        Jb_mdelta.setZero();
        pin::computeFrameJacobian(mdl_, mdl_data_, q_init - mp_dq * Ei, id_Fee_,
                                  pin::ReferenceFrame::WORLD, Jb_mdelta);
        double mp_mdelta =
            sqrt((Jb_mdelta * Jb_mdelta.transpose()).determinant());

        pin::Data::Matrix6x Jb_pdelta(6, n_q_);
        Jb_pdelta.setZero();
        pin::computeFrameJacobian(mdl_, mdl_data_, q_init + mp_dq * Ei, id_Fee_,
                                  pin::ReferenceFrame::WORLD, Jb_pdelta);
        double mp_pdelta =
            sqrt((Jb_pdelta * Jb_pdelta.transpose()).determinant());

        grad_manip[joint] = (mp_pdelta - mp_mdelta) / (2 * mp_dq);
      }

      if (verb_level_ >= 1)
        std::cout << "Manipulability gradient: " << grad_manip[0] << " "
                  << grad_manip[1] << " " << grad_manip[2] << " "
                  << grad_manip[3] << " " << grad_manip[4] << " "
                  << grad_manip[5] << " " << grad_manip[6] << " "
                  << grad_manip[7] << " " << grad_manip[8] << std::endl;

      par(n_q_ + 20) = dT;
      par(n_q_ + 21) = ca::DM(manip);
      par(ca::Slice(n_q_ + 22, 2 * n_q_ + 22)) = eig_to_casDM(grad_manip);
    }

    if (constrained_control_)
    {
      //? Use constrained control
      if (rcm_is_cost_)
      {
        //?  Using RCM error as cost
        arg_nlopt = {{"x0", ca::SX::zeros((n_q_, 1))},
                     {"p", par},
                     {"lbx", ca_qmin},
                     {"ubx", ca_qmax}};
      }
      else
      {
        //? Using RCM as constraint
        arg_nlopt = {{"x0", ca::SX::zeros((n_q_, 1))},
                     {"p", par},
                     {"lbx", ca_qmin},
                     {"ubx", ca_qmax},
                     {"lbg", ca::SX(0.0)},
                     {"ubg", ca::SX(rcm_error_max_)}};
      }
    }
    else
    {
      //? Use non-constrained control
      arg_nlopt = {{"x0", ca::SX::zeros((n_q_, 1))},
                   {"p", par},
                   {"lbx", ca_qmin},
                   {"ubx", ca_qmax}};
    }

    diff = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start_solve_time);
    time_left = max_time_ - diff.count() / 1000000.0;

    for (int it = 0;; it++)
    {
      if (verb_level_ >= 1)
        std::cout << "Starting it " << it << " with warm start "
                  << arg_nlopt["x0"] << std::endl;
      ca::DMDict res_nlopt = solver_(arg_nlopt);

      q_opt = res_nlopt["x"];
      f_opt = res_nlopt["f"];

      double opt_cost = static_cast<double>(f_opt);

      if (verb_level_ >= 1)
        std::cout << "\nIt solution: " << q_opt << std::endl;

      // Verify costs
      std::vector<ca::DM> xcost =
          ca_perr2_(std::vector<ca::DM>{ca::DM(ca_qin) + q_opt, ca_ps, ca_Rd});
      ca::DM c0 = ca::DM::sqrt(xcost[0]); // P cost
      ca::DM c1 = ca::DM::sqrt(xcost[1]); // R cost
      double p_err = static_cast<double>(c0);
      double r_err = static_cast<double>(c1);

      if (error_type_ == "log3")
        err_ee = sqrt(p_err * p_err + r_err * r_err);
      else if (error_type_ == "log6")
        err_ee = p_err;

      if (verb_level_ >= 1)
      {
        ca::DM c2 = ca::DM::mtimes(q_opt.T(), q_opt);
        std::cout << "\nAfer optimization:" << err_ee << std::endl;

        std::cout << "\tPose error : " << err_ee << std::endl;
        std::cout << "\tPosition error : " << p_err << std::endl;
        std::cout << "\tOrientation error : " << r_err << std::endl;
        std::cout << "\tCost0 : " << c0
                  << " Squared and Scaled: " << ca::DM(mu0_) * c0 * c0
                  << std::endl;
        std::cout << "\tCost1 : " << c1
                  << " Squared and Scaled: " << ca::DM(mu1_) * c1 * c1
                  << std::endl;
        std::cout << "\tCost2 : " << c2 << " Scaled: " << ca::DM(mu2_) * c2
                  << std::endl;
      }
      if (constrained_control_)
      {
        ca::DM c3, c4;
        c3 = ca::DM::sqrt(ca_ercm_(std::vector<ca::DM>{
            ca::DM(ca_qin) + q_opt, eig_to_casDM(B_p_Ftrocar_)})[0]);
        if (manipulability_enabled_)
        {
          ca::DM mp_err =
              manip - dT * ca::DM::mtimes(eig_to_casDM(grad_manip).T(), q_opt);
          c4 = ca::DM::mtimes(mp_err.T(), mp_err);
        }
        err_rcm = static_cast<double>(c3);

        if (verb_level_ >= 1)
        {
          std::cout << "Cost3 : " << c3 << " Scaled: " << ca::DM(mu3_) * c3
                    << std::endl;
          if (manipulability_enabled_)
            std::cout << "Cost4 : " << c4 << " Scaled: " << ca::DM(mu4_) * c4
                      << std::endl;
        }
      }

      if (constrained_control_)
      {
        if (error_type_ == "log3")
        {
          if (p_err < max_error_ && r_err < max_error_ &&
              err_rcm < rcm_error_max_)
          {
            success_ = true;

            break;
          }
        }
        else if (error_type_ == "log6")
        {
          if (p_err < max_error_ && err_rcm < rcm_error_max_)
          {
            success_ = true;
            break;
          }
        }
      }
      else
      {
        if (p_err < max_error_ && r_err < max_error_)
        {
          success_ = true;
          break;
        }
      }

      if (time_left < 0)
      {
        if (verb_level_ >= 1)
          std::cout << "NLO Aborted. Maximum time exceeded: " << time_left
                    << std::endl;
        break;
      }

      if (aborted_)
      {
        if (verb_level_ >= 1)
          std::cout << "NLO Aborted by other IK solver " << std::endl;
        break;
      }

      if (it >= max_iter_)
      {
        if (verb_level_ >= 1)
          std::cout << "NLO Aborted. Maximum number of iteration reached. "
                    << std::endl;
        break;
      }

      arg_nlopt = {
          {"x0", q_opt}, {"p", par}, {"lbx", ca_qmin}, {"ubx", ca_qmax}};

      diff = std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::high_resolution_clock::now() - start_solve_time);
      time_left = max_time_ - diff.count() / 1000000.0;
    }

    if (success_)
    {
      diff = std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::high_resolution_clock::now() - start_solve_time);
      q_sol = casDM_to_eig(ca::DM(ca_qin) + q_opt);
      std::vector<ca::DM> err_log3 = ca_elog3_ee_(
          std::vector<ca::DM>{ca::DM(ca_qin) + q_opt, ca_ps, ca_Rd});
      ca::DM log3_p = err_log3[0]; // P cost
      ca::DM log3_R = err_log3[1]; // R cost

      double err_log3_p = static_cast<double>(log3_p);
      double err_log3_R = static_cast<double>(log3_R);

      auto x_act = FK_(ca::DM(ca_qin) + q_opt);
      ca::DM p_act = x_act[0];
      ca::DM R_act = x_act[1];

      if (logging_enabled_)
      {
        fout_ << prob_id_ << "," << diff.count() << "," << err_rcm << ","
              << err_ee << "," << err_log3_p << "," << err_log3_R << ","
              << p_act(0) << "," << p_act(1) << "," << p_act(2) << ","
              << R_act(0, 0) << "," << R_act(1, 0) << "," << R_act(2, 0) << ","
              << R_act(0, 1) << "," << R_act(1, 1) << "," << R_act(2, 1)
              << "\n";
      }
      return 0;
    }
    else
    {
      if (verb_level_ >= 1)
        std::cout << "\nNLO: Warning: the iterative algorithm has not reached "
                     "convergence to the desired precision "
                  << std::endl;
      return 1;
    }
  }

  void NLO_CASADI::generate_nlsolver()
  {
    // Optimization Variable
    ca::SX q_delta = ca::SX::sym("qdelta", n_q_, 1);

    // Fixed parameters
    ca::SX par = ca::SX::sym("par", 2 * n_q_ + 22, 1);

    ca::SX q_in = par(ca::Slice(0, n_q_));
    ca::SX ca_ps = par(ca::Slice(n_q_, n_q_ + 3));
    ca::SX ca_Rd = ca::SX::reshape(par(ca::Slice(n_q_ + 3, n_q_ + 12)), 3, 3);
    ca::SX p_trocar = par(ca::Slice(n_q_ + 12, n_q_ + 15));
    ca::SX mu = par(ca::Slice(n_q_ + 15, n_q_ + 20));
    // ca::SX eps = par(28);
    ca::SX dT = par(n_q_ + 20);
    ca::SX manip = par(n_q_ + 21);
    ca::SX grad_manip = par(ca::Slice(n_q_ + 22, 2 * n_q_ + 22));

    // Optimization variable boundaries
    ca::SX ca_qmin = ca::SX::sym("qmin", n_q_, 1);
    ca::SX ca_qmax = ca::SX::sym("qmax", n_q_, 1);

    // Optimizer options
    ca::Dict opts;
    opts["verbose"] = false; // false
    opts["print_time"] = 0;
    opts["ipopt.linear_solver"] = nlo_linear_solver_;    //
    opts["ipopt.print_level"] = verb_level_;             // 0
    opts["ipopt.print_timing_statistics"] = time_stats_; //"no"
    // opts["ipopt.hessian_approximation"] = "limited-memory"; //"exact"
    opts["ipopt.warm_start_init_point"] = nlo_warm_start_; //"no"
    opts["ipopt.max_wall_time"] = max_time_;               //"no"
    if (nlo_concurrent_)
      opts["ipopt.max_iter"] = nlo_concurrent_iterations_;
    else
      opts["ipopt.max_iter"] = max_iter_;
    opts["ipopt.tol"] = max_error_; //"no"

    // Objective Function
    ca::SX obj;

    // Inequality constraints
    ca::SX cineq;

    // Nonlinear problem declaration
    ca::SXDict nlp;

    if (constrained_control_)
    {
      //? Use constrained control
      if (rcm_is_cost_)
      {
        //?  Using RCM error as cost
        if (verb_level_ >= 1)
          std::cout << "Using RCM as cost function" << std::endl;
        std::vector<ca::SX> xcost =
            ca_perr2_(std::vector<ca::SX>{q_in + q_delta, ca_ps, ca_Rd});

        ca::SX cost0 = xcost[0];
        ca::SX cost1 = xcost[1];

        ca::SX cost2 = ca::SX::mtimes(q_delta.T(), q_delta);
        ca::SX cost3 =
            ca_ercm_(std::vector<ca::SX>{q_in + q_delta, p_trocar})[0];

        if (manipulability_enabled_)
        {
          ca::SX mp_err = manip - dT * ca::SX::mtimes(grad_manip.T(), q_delta);
          ca::SX cost4 = ca::SX::mtimes(mp_err.T(), mp_err);

          obj = mu(0) * cost0 + mu(1) * cost1 + mu(2) * cost2 + mu(3) * cost3 +
                mu(4) * cost4;
        }
        else
        {
          obj = mu(0) * cost0 + mu(1) * cost1 + mu(2) * cost2 + mu(3) * cost3;
        }

        nlp = {{"x", q_delta}, {"p", par}, {"f", obj}};
      }
      else
      {
        //? Using RCM as constraint
        std::vector<ca::SX> xcost =
            ca_perr2_(std::vector<ca::SX>{q_in + q_delta, ca_ps, ca_Rd});
        ca::SX cost0 = xcost[0];
        ca::SX cost1 = xcost[1];
        ca::SX cost2 = ca::SX::mtimes(q_delta.T(), q_delta);
        obj = mu(0) * cost0 + mu(1) * cost1 + mu(2) * cost2;

        cineq = ca_ercm_(std::vector<ca::SX>{q_in + q_delta, p_trocar})[0];

        nlp = {{"x", q_delta}, {"p", par}, {"f", obj}, {"g", cineq}};
      }
    }
    else
    {
      //? Use non-constrained control
      if (verb_level_ >= 1)
        std::cout << "No constrained motion" << std::endl;

      std::vector<ca::SX> xcost =
          ca_perr2_(std::vector<ca::SX>{q_in + q_delta, ca_ps, ca_Rd});
      ca::SX cost0 = xcost[0];
      ca::SX cost1 = xcost[1];
      ca::SX cost2 = ca::SX::mtimes(q_delta.T(), q_delta);
      obj = mu(0) * cost0 + mu(1) * cost1 + mu(2) * cost2;

      nlp = {{"x", q_delta}, {"p", par}, {"f", obj}};
    }

    solver_ = nlpsol("nlpsol", "ipopt", nlp, opts);
  }

  void NLO_CASADI::generate_ca_EE_error()
  {
    if (verb_level_ >= 1)
      std::cout << "Generating Casadi EE error function" << std::endl;
    //* Generate perr2 = perr.T*perr Casadi function
    // Inputs
    ca::SX R_des = ca::SX::sym("R_act", 3, 3);
    ca::SX p_des = ca::SX::sym("p_act", 3, 1);
    ca::SX q_init = ca::SX::sym("q_init", n_q_, 1);

    ca::SX p_act = FK_(q_init)[0];
    ca::SX R_act = FK_(q_init)[1];

    ca::SX p_e =
        p_des - ca::SX::mtimes(R_des, ca::SX::mtimes(R_act.T(), p_act));
    ca::SX R_e = ca::SX::mtimes(R_des, R_act.T());

    ca::SX err;

    ca::SX p_error;
    ca::SX R_error;
    ca::SX p_error2;
    ca::SX R_error2;

    if (error_type_ == "log6")
    {
      //? Using log6
      std::vector<ca::SX> err_tmp = ca_log6_(std::vector<ca::SX>{p_e, R_e});
      p_error2 = ca::SX::mtimes(err_tmp[0].T(), err_tmp[0]);
      R_error2 = ca::SX(0.0);
    }
    else if (error_type_ == "log3")
    {
      //? Using log3
      p_error = p_des - p_act;
      p_error2 = ca::SX::mtimes(p_error.T(), p_error);
      R_error = ca_log3_(std::vector<ca::SX>{R_e})[0];
      R_error2 = ca::SX::mtimes(R_error.T(), R_error);
    }
    else
    {
      //? Using only p error
      p_error = p_des - p_act;
      p_error2 = ca::SX::mtimes(p_error.T(), p_error);
      R_error2 = ca::SX(0.0);
    }

    ca_perr2_ =
        ca::Function("p_err2", {q_init, p_des, R_des}, {p_error2, R_error2},
                     {"q_init", "pd", "Rd"}, {"p_err2", "R_err2"});
  }

  void NLO_CASADI::generate_ca_log3_EE_error()
  {
    if (verb_level_ >= 1)
      std::cout << "Generating Casadi Log3 EE error function" << std::endl;
    //* Generate perr2 = perr.T*perr Casadi function
    // Inputs
    ca::SX R_des = ca::SX::sym("R_act", 3, 3);
    ca::SX p_des = ca::SX::sym("p_act", 3, 1);
    ca::SX q_init = ca::SX::sym("q_init", n_q_, 1);

    ca::SX p_act = FK_(q_init)[0];
    ca::SX R_act = FK_(q_init)[1];

    ca::SX R_e = ca::SX::mtimes(R_des, R_act.T());

    ca::SX p_err_vec;
    ca::SX R_err_vec;
    ca::SX p_error;
    ca::SX R_error;

    p_err_vec = p_des - p_act;
    p_error = ca::SX::sqrt(ca::SX::mtimes(p_err_vec.T(), p_err_vec));

    R_err_vec = ca_log3_(std::vector<ca::SX>{R_e})[0];
    R_error = ca::SX::sqrt(ca::SX::mtimes(R_err_vec.T(), R_err_vec));

    ca_elog3_ee_ =
        ca::Function("err_log3", {q_init, p_des, R_des}, {p_error, R_error},
                     {"q_init", "pd", "Rd"}, {"p_error", "R_error"});
  }

  void NLO_CASADI::generate_ca_RCM_error()
  {
    // Generating RCM error CASADI function
    if (verb_level_ >= 1)
      std::cout << "Generating Casadi RCM error function" << std::endl;
    ca::SX qvec = ca::SX::sym("qvec", n_q_, 1);
    ca::SX p_trocar = ca::SX::sym("p_trocar", 3, 1);

    ca::SX p_prercm = FK_Fpre_(qvec)[0];
    ca::SX p_postrcm = FK_Fpost_(qvec)[0];
    ca::SX p_ee = FK_(qvec)[0];
    ca::SX R_prercm = FK_Fpre_(qvec)[1];
    ca::SX R_postrcm = FK_Fpost_(qvec)[1];
    ca::SX R_ee = FK_(qvec)[1];

    ca::SX ca_ps = p_postrcm - p_prercm;
    ca::SX ca_pr = p_trocar - p_prercm;

    // RCM error
    ca::SX e_rcm =
        ca_pr - ca::SX::mtimes((ca::SX::mtimes(ca_pr.T(), ca_ps)), ca_ps) /
                    (ca::SX::mtimes(ca_ps.T(), ca_ps));
    ca::SX e_rcm2 = ca::SX::mtimes(e_rcm.T(), e_rcm);

    ca_ercm_ = ca::Function("e_rcm", {qvec, p_trocar}, {e_rcm2},
                            {"q", "p_trocar"}, {"e_rcm"});
  }

  void NLO_CASADI::generate_ca_log3()
  {
    if (verb_level_ >= 1)
      std::cout << "Generating Casadi log3 function" << std::endl;
    ca::SX tolerance = 1e-8;

    ca::SX R = ca::SX::sym("R", 3, 3);
    ca::SX omega = ca::SX::sym("omega", 3, 1);
    ca::SX val = (ca::SX::trace(R) - ca::SX(1)) / ca::SX(2);
    val = ca::SX::if_else(val > ca::SX(1), ca::SX(1),
                          ca::SX::if_else(val < ca::SX(-1), ca::SX(-1), val));
    ca::SX theta = ca::SX::acos(val);
    ca::SX stheta = ca::SX::sin(theta);
    ca::SX tr = ca::SX::if_else(theta < tolerance, ca::SX::zeros((3, 3)),
                                (R - R.T()) * theta / (ca::SX(2) * stheta));
    omega = ca::SX::inv_skew(tr);
    ca_log3_ =
        ca::Function("ca_log3", {R}, {omega, theta}, {"R"}, {"w", "theta"});
  }

  void NLO_CASADI::generate_ca_log6()
  {
    if (verb_level_ >= 1)
      std::cout << "Generating Casadi log6 function" << std::endl;
    ca::SX tolerance = 1e-8;
    ca::SX tolerance2 = 1e-16;

    ca::SX tau = ca::SX::sym("tau", 6, 1);
    ca::SX R = ca::SX::sym("R", 3, 3);
    ca::SX p = ca::SX::sym("p", 3, 1);

    std::vector<ca::SX> log_res = ca_log3_(R);
    ca::SX omega = log_res[0];
    ca::SX theta = log_res[1];

    ca::SX stheta = ca::SX::sin(theta);
    ca::SX ctheta = ca::SX::cos(theta);

    ca::SX A_inv = ca::SX::if_else(
        ca::SX::mtimes(p.T(), p) < tolerance2, ca::SX::zeros((3, 3)),
        ca::SX::if_else(
            theta < tolerance, ca::SX::eye(3),
            ca::SX::eye(3) - ca::SX::skew(omega) / ca::SX(2) +
                (ca::SX(2) * stheta - theta * (ca::SX(1) + ctheta)) *
                    (ca::SX::mtimes(ca::SX::skew(omega), ca::SX::skew(omega))) /
                    (ca::SX(2) * (ca::SX::pow(theta, 2)) * stheta)));

    ca::SX v = ca::SX::mtimes(A_inv, p);

    tau(ca::Slice(0, 3)) = v;
    tau(ca::Slice(3, 6)) = omega;

    ca_log6_ = ca::Function("ca_log6", {p, R}, {tau}, {"p", "R"}, {"tau"});
  }

  //* Casadi-Eigen conversion functions

  ca::DM NLO_CASADI::eig_to_casDM(const VectorXd &eig)
  {
    auto dm = casadi::DM(casadi::Sparsity::dense(eig.size()));
    for (int i = 0; i < eig.size(); i++)
    {
      dm(i) = eig(i);
    }
    return dm;
  }

  ca::DM NLO_CASADI::eigmat_to_casDM(const MatrixXd &eig)
  {
    casadi::DM dm = casadi::DM(casadi::Sparsity::dense(eig.rows(), eig.cols()));
    std::copy(eig.data(), eig.data() + eig.size(), dm.ptr());
    return dm;
  }

  MatrixXd NLO_CASADI::casDM_to_eig(const casadi::DM &cas)
  {
    auto vector_x = static_cast<std::vector<double>>(cas);
    MatrixXd eig = MatrixXd::Zero(cas.size1(), cas.size2());

    for (int i = 0; i < cas.size1(); i++)
    {
      for (int j = 0; j < cas.size2(); j++)
      {
        eig(i, j) = vector_x[i + j * cas.size2()];
      }
    }
    return eig;
  }

} // namespace coiks

#endif
