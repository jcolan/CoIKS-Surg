/******************************************************************************
# unconstrained_benchmark.cpp:  Unconstrained benchmark for COIKS             #
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
#include <coiks/ik_solver.hpp>
#include <csignal>
#include <string>

// ROS
#include <ros/package.h>
#include <sensor_msgs/JointState.h>

// Orocos KDL
#include <kdl/chain.hpp>
#include <kdl/chainfksolver.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainiksolverpos_nr_jl.hpp>
#include <kdl/chainiksolvervel_pinv.hpp>
#include <kdl/chainjnttojacsolver.hpp>
#include <kdl/frames.hpp>
#include <kdl/jacobian.hpp>
#include <kdl/jntarray.hpp>
#include <kdl/velocityprofile_spline.hpp>
#include <kdl/velocityprofile_trap.hpp>

// Trac_IK
#include <trac_ik/trac_ik.hpp>

// KDL Parser
#include <kdl_parser/kdl_parser.hpp>

using namespace coiks;
namespace pin = pinocchio;

class BenchmarkKDL : public IkSolver
{
public:
  BenchmarkKDL(const std::string &_urdf_file, const std::string &_base_link,
               const std::string &_ee_link, const std::string &_ik_solver,
               double _max_time, double _max_error, int _max_iter, double _dt)
      : initialized_(false), max_error_(_max_error), max_iter_(_max_iter),
        urdf_file_(_urdf_file), base_link_(_base_link), ee_link_(_ee_link),
        dt_(_dt), solver_name_("kdl")
  {
    initialize_kdl();
  }

  ~BenchmarkKDL() {}

  bool initialize_kdl()
  {
    std::cout << "Initializing KDL with Max. Error: " << max_error_
              << " Max. It.:" << max_iter_ << " Delta T:" << dt_ << std::endl;

    double maxtime = 0.005;

    // Parsing URDF
    if (!kdl_parser::treeFromFile(urdf_file_, kdl_tree_))
    {
      ROS_ERROR("Failed to construct kdl tree");
      return false;
    }
    bool exit_value = kdl_tree_.getChain(base_link_, ee_link_, kdl_chain_);
    // Resize variables
    n_joints_ = kdl_chain_.getNrOfJoints();
    qtmp_.resize(kdl_chain_.getNrOfJoints());
    nominal_.resize(kdl_chain_.getNrOfJoints());
    ll_.resize(kdl_chain_.getNrOfJoints());
    ul_.resize(kdl_chain_.getNrOfJoints());

    //   Load the urdf model
    pin::Model model_;
    pin::urdf::buildModel(urdf_file_, model_);

    // Getting Joints Limits
    q_ul_ = model_.upperPositionLimit;
    q_ll_ = model_.lowerPositionLimit;

    // Storing Joint limits
    for (int i = 0; i < kdl_chain_.getNrOfJoints(); i++)
    {
      ll_.data(i) = q_ll_[i];
      ul_.data(i) = q_ul_[i];
    }

    assert(kdl_chain_.getNrOfJoints() == ll_.data.size());
    assert(kdl_chain_.getNrOfJoints() == ul_.data.size());

    kdl_fk_solver_.reset(new KDL::ChainFkSolverPos_recursive(kdl_chain_));
    kdl_vik_solver_.reset(new KDL::ChainIkSolverVel_pinv(kdl_chain_));
    kdl_ik_solver_.reset(new KDL::ChainIkSolverPos_NR_JL(
        kdl_chain_, ll_, ul_, *kdl_fk_solver_, *kdl_vik_solver_, max_iter_,
        max_error_));

    // Initialize nominal vector
    for (uint j = 0; j < nominal_.data.size(); j++)
    {
      nominal_(j) = (ll_(j) + ul_(j)) / 2.0;
    }
    std::cout << "KDL initialized" << std::endl;
    initialized_ = true;

    return true;
  }

  // * KDL solver
  int IkSolve(const VectorXd q_init, const pin::SE3 &x_Fee_d, VectorXd &q_out)
  {
    bool success = false;
    int rc;
    KDL::JntArray qd(n_joints_);
    KDL::Frame ee;

    Affine3d Tdes;
    Tdes.linear() = x_Fee_d.rotation();
    Tdes.translation() = x_Fee_d.translation();

    tf::transformEigenToKDL(Tdes, ee);

    rc = kdl_ik_solver_->CartToJnt(nominal_, ee, qd);

    if (rc >= 0)
    {
      q_out = VectorXd::Map(&qd.data[0], qd.data.size());
      success = true;
      return 0;
    }
    else
    {
      success = false;
      return 1;
    }
  }

  int get_n_joints() { return n_joints_; }
  std::string get_solver_name() { return solver_name_; }

private:
  // Temporary variables for KDL
  KDL::Tree kdl_tree_;
  KDL::Chain kdl_chain_;

  KDL::JntArray nominal_;
  KDL::JntArray qtmp_;
  KDL::JntArray ll_, ul_;
  KDL::Frame xtmp_;
  KDL::Jacobian Jtmp_;

  KDL::Twist xdot_temp_;
  KDL::JntArray qdot_tmp_;

  // KDL
  std::unique_ptr<KDL::ChainFkSolverPos_recursive> kdl_fk_solver_;
  std::unique_ptr<KDL::ChainIkSolverVel_pinv> kdl_vik_solver_;
  std::unique_ptr<KDL::ChainIkSolverPos_NR_JL> kdl_ik_solver_;

  std::string urdf_file_;

  std::string base_link_;
  std::string ee_link_;
  VectorXd q_ul_;
  VectorXd q_ll_;

  int n_joints_;
  std::string solver_name_;
  int max_iter_;
  double max_error_;

  bool initialized_;
  double dt_;
};

class BenchmarkTRACIK : public IkSolver
{
public:
  BenchmarkTRACIK(const std::string &_urdf_file, const std::string &_base_link,
                  const std::string &_ee_link, const std::string &_ik_solver,
                  double _max_time, double _max_error, int _max_iter,
                  double _dt)
      : initialized_(false), max_error_(_max_error), max_time_(_max_time),
        urdf_file_(_urdf_file), base_link_(_base_link), ee_link_(_ee_link),
        dt_(_dt), solver_name_("trac_ik")
  {
    initialize_tracik();
  }

  ~BenchmarkTRACIK() {}

  bool initialize_tracik()
  {
    std::cout << "Initializing TRACIK with Max. Error: " << max_error_
              << " Max. Time:" << max_time_ << " Delta T:" << dt_ << std::endl;

    // Parsing URDF
    if (!kdl_parser::treeFromFile(urdf_file_, kdl_tree_))
    {
      ROS_ERROR("Failed to construct kdl tree");
      return false;
    }
    bool exit_value = kdl_tree_.getChain(base_link_, ee_link_, kdl_chain_);

    // Resize variables
    n_joints_ = kdl_chain_.getNrOfJoints();
    qtmp_.resize(kdl_chain_.getNrOfJoints());
    nominal_.resize(kdl_chain_.getNrOfJoints());
    ll_.resize(kdl_chain_.getNrOfJoints());
    ul_.resize(kdl_chain_.getNrOfJoints());

    //   Load the urdf model
    pin::Model model_;
    pin::urdf::buildModel(urdf_file_, model_);

    // Getting Joints Limits
    q_ul_ = model_.upperPositionLimit;
    q_ll_ = model_.lowerPositionLimit;

    // Storing Joint limits
    for (int i = 0; i < kdl_chain_.getNrOfJoints(); i++)
    {
      ll_.data(i) = q_ll_[i];
      ul_.data(i) = q_ul_[i];
    }

    assert(kdl_chain_.getNrOfJoints() == ll_.data.size());
    assert(kdl_chain_.getNrOfJoints() == ul_.data.size());

    // Initialize Trac-IK
    tracik_solver_.reset(new TRAC_IK::TRAC_IK(kdl_chain_, ll_, ul_, max_time_,
                                              max_error_, TRAC_IK::Speed));

    // Initialize nominal vector
    for (uint j = 0; j < nominal_.data.size(); j++)
    {
      nominal_(j) = (ll_(j) + ul_(j)) / 2.0;
    }
    std::cout << "TRACIK initialized" << std::endl;

    return true;
  }

  // * TRAC-IK solver
  int IkSolve(const VectorXd q_init, const pin::SE3 &x_Fee_d, VectorXd &q_out)
  {
    bool success = false;
    int rc;
    KDL::JntArray qd(n_joints_);
    KDL::Frame ee;

    Affine3d Tdes;
    Tdes.linear() = x_Fee_d.rotation();
    Tdes.translation() = x_Fee_d.translation();

    tf::transformEigenToKDL(Tdes, ee);

    rc = tracik_solver_->CartToJnt(nominal_, ee, qd);

    if (rc >= 0)
    {
      q_out = VectorXd::Map(&qd.data[0], qd.data.size());

      success = true;
      return 0;
    }
    else
    {
      success = false;
      return 1;
    }
  }

  int get_n_joints() { return n_joints_; }
  std::string get_solver_name() { return solver_name_; }

private:
  std::unique_ptr<TRAC_IK::TRAC_IK> tracik_solver_;

  std::string urdf_file_;

  std::string base_link_;
  std::string ee_link_;
  VectorXd q_ul_;
  VectorXd q_ll_;

  int n_joints_;
  std::string solver_name_;
  double max_error_;
  double max_time_;

  // Temporary variables for KDL
  KDL::Tree kdl_tree_;
  KDL::Chain kdl_chain_;

  KDL::JntArray nominal_;
  KDL::JntArray qtmp_;
  KDL::JntArray ll_, ul_;

  bool initialized_;
  double dt_;
};

double fRand(double min, double max)
{
  double f = (double)rand() / RAND_MAX;
  return min + f * (max - min);
}

void solveIkFromRandomList(std::string urdf_file_, int num_samples,
                           IkSolver *ik_solver, std::string ee_link_,
                           bool print_all = false)
{
  double total_time = 0;
  double total_time_it = 0;
  double success_time_it = 0;
  uint n_success = 0;

  //   Pinocchio variables
  pin::Model model_;
  pin::Data mdl_data_;
  pin::FrameIndex ee_id_;

  VectorXd q_ul_;
  VectorXd q_ll_;
  VectorXd q_sol_;

  std::vector<double> errors_;
  int n_joints_;

  std::fstream fout;
  int prob_id = 0;

  //   Load the urdf model
  pin::urdf::buildModel(urdf_file_, model_);
  ee_id_ = model_.getFrameId(ee_link_);

  n_joints_ = model_.nq;
  q_sol_.resize(n_joints_);
  errors_.clear();

  // Getting Joints Limits
  q_ul_ = model_.upperPositionLimit;
  q_ll_ = model_.lowerPositionLimit;

  std::cout << "Solving IK with " << ik_solver->get_solver_name() << " for "
            << num_samples << " random configurations for link " << ee_link_
            << std::endl;

  // Create desired number of valid, random joint configurations
  std::vector<VectorXd> JointList;
  VectorXd q(ik_solver->get_n_joints());

  for (uint i = 0; i < num_samples; i++)
  {
    for (uint j = 0; j < q_ll_.size(); j++)
    {
      q(j) = fRand(q_ll_(j), q_ul_(j));
    }
    JointList.push_back(q);
  }

  pin::Data mdl_data(model_);
  VectorXd q_init = pin::neutral(model_);
  q_init = pin::randomConfiguration(model_);

  VectorXd q_sol = pin::neutral(model_);

  time_t currentTime;
  struct tm *localTime;

  time(&currentTime); // Get the current time
  localTime = localtime(&currentTime);
  if (ik_solver->get_solver_name() == "kdl")
  {
    fout.open("/home/colan/kdl_" + std::to_string(localTime->tm_mday) +
                  std::to_string(localTime->tm_hour) +
                  std::to_string(localTime->tm_min) +
                  std::to_string(localTime->tm_sec) + ".csv",
              std::ios::out | std::ios::app);
    fout << "idx,time\n";
  }
  else if (ik_solver->get_solver_name() == "trac_ik")
  {
    fout.open("/home/colan/tracik_" + std::to_string(localTime->tm_mday) +
                  std::to_string(localTime->tm_hour) +
                  std::to_string(localTime->tm_min) +
                  std::to_string(localTime->tm_sec) + ".csv",
              std::ios::out | std::ios::app);
    fout << "idx,time\n";
  }

  prob_id = 0;

  auto start_cb_time = std::chrono::high_resolution_clock::now();
  auto start_it_time = std::chrono::high_resolution_clock::now();
  auto stop_it_time = std::chrono::high_resolution_clock::now();

  for (uint i = 0; i < num_samples; i++)
  {
    prob_id++;
    if (print_all)
      std::cout << "\n[Prob " << i
                << "] Solving for Joints: " << JointList[i].transpose()
                << std::endl;
    int res = 1;
    pin::forwardKinematics(model_, mdl_data, JointList[i]);
    pin::updateFramePlacements(model_, mdl_data);
    pin::SE3 x_des = mdl_data.oMf[ee_id_];
    if (print_all)
      std::cout << "[Prob " << i
                << "] Solving for Pos: " << x_des.translation().transpose()
                << std::endl;
    start_it_time = std::chrono::high_resolution_clock::now();

    // Call IK Solver
    res = ik_solver->IkSolve(q_init, x_des, q_sol);

    stop_it_time = std::chrono::high_resolution_clock::now();

    if (res && print_all)
      ROS_WARN("Solution not found");

    auto duration_it = std::chrono::duration_cast<std::chrono::microseconds>(
        stop_it_time - start_it_time);
    if (print_all)
    {
      std::cout << "Time: " << duration_it.count() << " [us]" << std::endl;
      if (!res)
      {
        std::cout << "Solution: " << q_sol.transpose() << std::endl;
      }
    }
    total_time_it += duration_it.count();

    if (!res)
    {
      success_time_it += duration_it.count();
      n_success++;

      if (ik_solver->get_solver_name() == "kdl")
      {
        fout << prob_id << "," << duration_it.count() << "\n";
      }
      else if (ik_solver->get_solver_name() == "trac_ik")
      {
        fout << prob_id << "," << duration_it.count() << "\n";
      }
    }
  }

  auto stop_cb_time = std::chrono::high_resolution_clock::now();
  auto duration_cb = std::chrono::duration_cast<std::chrono::microseconds>(
      stop_cb_time - start_cb_time);

  total_time = duration_cb.count();

  std::cout << "------------------------------" << std::endl;
  std::cout << "Summary:" << std::endl;

  std::cout << ik_solver->get_solver_name() << " found " << n_success
            << " solutions (" << 100.0 * n_success / num_samples
            << "\%) with a total average of " << total_time / num_samples
            << " usec/config. Solving average of "
            << total_time_it / num_samples << " usec/config."
            << "Success solving average of " << success_time_it / n_success
            << " usec/config." << std::endl;
  if (ik_solver->get_solver_name() == "codcs")
  {
    ik_solver->printStatistics();
  }
  std::cout << "------------------------------" << std::endl;

  if (ik_solver->get_solver_name() == "trac_ik" ||
      ik_solver->get_solver_name() == "kdl")
    fout.close();

  return;
}

bool kill_process = false;
void SigIntHandler(int signal)
{
  kill_process = true;
  ROS_INFO_STREAM("SHUTDOWN SIGNAL RECEIVED");
}

int main(int argc, char **argv)
{
  ROS_INFO("Unscontrained IK Benchmarking for COIKS, TRAC-IK and KDL");
  ros::init(argc, argv, "coiks_benchmark");
  ros::NodeHandle nh;
  std::signal(SIGINT, SigIntHandler);

  // ROS parameters
  std::string ik_solver;
  int max_iter;
  int n_random_config;
  double max_time;
  double max_error;
  double dt;
  int print_all;

  if (!nh.getParam("ik_solver", ik_solver))
  {
    ik_solver = "tracik";
  }
  std::string ee_link_name;
  if (!nh.getParam("ee_link_name", ee_link_name))
  {
    ee_link_name = "ee_link";
  }
  if (!nh.getParam("max_iter", max_iter))
  {
    max_iter = 10000;
  }
  if (!nh.getParam("max_time", max_time))
  {
    max_time = 5e-3;
  }
  if (!nh.getParam("max_error", max_error))
  {
    max_error = 1e-5;
  }
  if (!nh.getParam("delta_integration", dt))
  {
    dt = 1.0;
  }
  std::string error_type;
  if (!nh.getParam("error_type", error_type))
  {
    error_type = "log6";
  }
  bool manipulability_enabled;
  if (!nh.getParam("manipulability_enabled", manipulability_enabled))
  {
    manipulability_enabled = false;
  }

  //* Constraints variables
  double rcm_error_max;
  if (!nh.getParam("rcm_error_max", rcm_error_max))
  {
    rcm_error_max = 5e-4;
  }
  bool constrained_control;
  if (!nh.getParam("constrained_control", constrained_control))
  {
    constrained_control = false;
  }
  bool rcm_is_cost;
  if (!nh.getParam("rcm_is_cost", rcm_is_cost))
  {
    rcm_is_cost = true;
  }
  Vector3d trocar_pos;
  double trocar_x;
  double trocar_y;
  double trocar_z;
  if (!nh.getParam("trocar_x", trocar_x))
  {
    trocar_x = 0.426;
  }
  if (!nh.getParam("trocar_y", trocar_y))
  {
    trocar_y = 0.0;
  }
  if (!nh.getParam("trocar_z", trocar_z))
  {
    trocar_z = 0.485;
  }
  trocar_pos[0] = trocar_x;
  trocar_pos[1] = trocar_y;
  trocar_pos[2] = trocar_z;

  //* Esperiment variables
  int robot_id;
  if (!nh.getParam("robot_id", robot_id))
  {
    robot_id = 0;
  }
  if (!nh.getParam("n_random_config", n_random_config))
  {
    n_random_config = 100;
  }

  //* INVJ Variables
  double invj_Ke1;
  if (!nh.getParam("invj_Ke1", invj_Ke1))
  {
    invj_Ke1 = 1.0;
  }

  double invj_Ke2;
  if (!nh.getParam("invj_Ke2", invj_Ke2))
  {
    invj_Ke2 = 1.0;
  }

  //* NLO Variables
  std::string nlo_linear_solver;
  if (!nh.getParam("nlo_linear_solver", nlo_linear_solver))
  {
    nlo_linear_solver = "ma57";
  }

  double mu0;
  if (!nh.getParam("nlo_mu0", mu0))
  {
    mu0 = 1.0;
  }

  double mu1;
  if (!nh.getParam("nlo_mu1", mu1))
  {
    mu1 = 0.005;
  }

  double mu2;
  if (!nh.getParam("nlo_mu2", mu2))
  {
    mu2 = 0.001;
  }

  double mu3;
  if (!nh.getParam("nlo_mu3", mu3))
  {
    mu3 = 100.0;
  }

  double mu4;
  if (!nh.getParam("nlo_mu4", mu4))
  {
    mu4 = 0.01;
  }

  bool nlo_concurrent;
  if (!nh.getParam("nlo_concurrent", nlo_concurrent))
  {
    nlo_concurrent = false;
  }
  std::string nlo_error_type;
  if (!nh.getParam("nlo_error_type", nlo_error_type))
  {
    nlo_error_type = "log3";
  }
  int nlo_concurrent_iterations;
  if (!nh.getParam("nlo_concurrent_iterations", nlo_concurrent_iterations))
  {
    nlo_concurrent_iterations = 5;
  }
  std::string nlo_warm_start;
  if (!nh.getParam("nlo_warm_start", nlo_warm_start))
  {
    nlo_warm_start = "yes";
  }

  //* HQP variables
  double hqp_Ke1;
  if (!nh.getParam("hqp_Ke1", hqp_Ke1))
  {
    hqp_Ke1 = 1.0;
  }
  double hqp_Ke2;
  if (!nh.getParam("hqp_Ke2", hqp_Ke2))
  {
    hqp_Ke2 = 1.0;
  }
  double hqp_Kd1;
  if (!nh.getParam("hqp_Kd1", hqp_Kd1))
  {
    hqp_Kd1 = 0.00001;
  }
  double hqp_Kd2;
  if (!nh.getParam("hqp_Kd2", hqp_Kd2))
  {
    hqp_Kd2 = 0.00001;
  }
  bool hqp_warm_start;
  if (!nh.getParam("hqp_warm_start", hqp_warm_start))
  {
    hqp_warm_start = true;
  }

  //* Printing variables
  std::string time_stats;
  if (!nh.getParam("solv_time_stats", time_stats))
  {
    time_stats = "no";
  }
  int verb_level;
  if (!nh.getParam("solv_verb_level", verb_level))
  {
    verb_level = 0;
  }
  if (!nh.getParam("print_all", print_all))
  {
    print_all = 0;
  }
  bool logging_enabled;
  if (!nh.getParam("logging_enabled", logging_enabled))
  {
    logging_enabled = false;
  }
  std::string log_path;
  if (!nh.getParam("log_path", log_path))
  {
    log_path = "";
  }

  bool tracik_enable;
  if (!nh.getParam("tracik_enable", tracik_enable))
  {
    tracik_enable = true;
  }

  bool kdl_enable;
  if (!nh.getParam("kdl_enable", kdl_enable))
  {
    kdl_enable = false;
  }

  // Setting up URDF path
  std::string pkg_path = ros::package::getPath("coiks");
  // std::string urdf_path = pkg_path + std::string("/urdf/") +
  // "smart_arm_r.urdf";

  std::string urdf_path;

  switch (robot_id)
  {
  case 0:
    urdf_path = pkg_path + std::string("/urdf/") + "robot1_endoscope.urdf";
    break;

  case 1:
    urdf_path = pkg_path + std::string("/urdf/") + "robot2_openrst.urdf";
    break;

  case 2:
    urdf_path = pkg_path + std::string("/urdf/") + "robot3_hyperrst.urdf";
    break;

  default:
    ROS_ERROR("Kinematic Tree ID not recognized.");
    return -1;
  }
  ROS_INFO_STREAM("Using URDF found in: " << urdf_path);

  // Creating solver options class
  SolverOptions so;

  so.error_type_ = error_type;
  so.constrained_control_ = constrained_control;
  so.rcm_error_max_ = rcm_error_max;
  so.rcm_is_cost_ = rcm_is_cost;
  so.manipulability_enabled_ = manipulability_enabled;

  so.invj_Ke1_ = invj_Ke1;
  so.invj_Ke2_ = invj_Ke2;

  so.nlo_linear_solver_ = nlo_linear_solver;
  so.cost_coeff_.push_back(mu0);
  so.cost_coeff_.push_back(mu1);
  so.cost_coeff_.push_back(mu2);
  so.cost_coeff_.push_back(mu3);
  so.cost_coeff_.push_back(mu4);
  so.nlo_concurrent_ = nlo_concurrent;
  so.nlo_error_type_ = nlo_error_type;
  so.nlo_concurrent_iterations_ = nlo_concurrent_iterations;
  so.nlo_warm_start_ = nlo_warm_start;

  so.hqp_Ke1_ = hqp_Ke1;
  so.hqp_Ke2_ = hqp_Ke2;
  so.hqp_Kd1_ = hqp_Kd1;
  so.hqp_Kd2_ = hqp_Kd2;
  so.hqp_warm_start_ = hqp_warm_start;

  so.time_stats_ = time_stats;
  so.verb_level_ = verb_level;
  so.logging_enabled_ = logging_enabled;
  so.log_path_ = log_path;

  ROS_INFO_STREAM("Error type: " << so.error_type_);

  // Initialiing KDL
  ROS_INFO("Starting KDL");
  BenchmarkKDL kdl_ik(urdf_path, "base_link", "link_ee", ik_solver, max_time,
                      max_error, max_iter, dt);

  // Initialiing TRAC-IK
  ROS_INFO("Starting TRAC-IK");
  BenchmarkTRACIK trac_ik(urdf_path, "base_link", "link_ee", ik_solver,
                          max_time, max_error, max_iter, dt);

  // Initialiing COIKS
  ROS_INFO("Starting CODCS-IK");
  COIKS coiks(urdf_path, "base_link", "link_ee", ik_solver, so, max_time,
              max_error, max_iter, dt);

  if (kdl_enable)
  {
    ROS_WARN("Running random configurations for KDL");
    solveIkFromRandomList(urdf_path, n_random_config, &kdl_ik, "link_ee",
                          (print_all & 1));
  }

  if (tracik_enable)
  {
    ROS_WARN("Running random configurations for TRAC-IK");
    solveIkFromRandomList(urdf_path, n_random_config, &trac_ik, "link_ee",
                          (print_all & 2) >> 1);
  }

  ROS_WARN("Running random configurations for COIKS");
  solveIkFromRandomList(urdf_path, n_random_config, &coiks, "link_ee",
                        (print_all & 4) >> 2);
  ROS_INFO("Benchmark finished");

  return 0;
}
