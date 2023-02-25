/******************************************************************************
# constrained_benchmark.cpp:  Constrained benchmark for COIKS                 #
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

using namespace coiks;
namespace pin = pinocchio;

double fRand(double min, double max)
{
  double f = (double)rand() / RAND_MAX;
  return min + f * (max - min);
}

void solveIkFromGivenList(ros::NodeHandle &nh, std::string urdf_file_,
                          IkSolver *ik_solver, std::string ee_link_,
                          std::vector<pin::SE3> path, int robot_id,
                          bool print_all = false)
{
  int num_samples = path.size();
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

  ros::Publisher pub_joint_sol =
      nh.advertise<sensor_msgs::JointState>("/sim/joint/cmd", 1);

  sensor_msgs::JointState pub_joint_cmd_msg;

  std::vector<double> errors_;
  int n_joints_;

  //   Load the urdf model
  pin::urdf::buildModel(urdf_file_, model_);
  ee_id_ = model_.getFrameId(ee_link_);

  n_joints_ = model_.nq;
  errors_.clear();

  // Getting Joints Limits
  q_ul_ = model_.upperPositionLimit;
  q_ll_ = model_.lowerPositionLimit;

  std::cout << "Solving IK with " << ik_solver->get_solver_name() << " for "
            << num_samples << " samples in given path for link " << ee_link_
            << std::endl;

  pin::Data mdl_data(model_);

  VectorXd q_init = pin::randomConfiguration(model_);

  if (robot_id == 0)
  {
    q_init << 0.0, 0.0, 1.75, 0.0, -0.79, 1.57;
    pub_joint_cmd_msg.name.resize(6);
    pub_joint_cmd_msg.position.resize(6);
    pub_joint_cmd_msg.name = {"joint1", "joint2", "joint3",
                              "joint4", "joint5", "joint6"};
  }
  else if (robot_id == 1)
  {
    q_init << 0.0, -0.17, 0.0, 1.31, 0.0, 1.57, 0.0, 0.0, 0.0;
    pub_joint_cmd_msg.name.resize(9);
    pub_joint_cmd_msg.position.resize(9);
    pub_joint_cmd_msg.name = {"joint1", "joint2", "joint3", "joint4", "joint5",
                              "joint6", "joint7", "joint8", "joint9"};
  }
  else if (robot_id == 2)
  {
    q_init << 0.0, -0.17, 0.0, 1.31, 0.0, 1.57, 0.0, 0.0, 0.0, 0.0, 0.0;
    pub_joint_cmd_msg.name.resize(11);
    pub_joint_cmd_msg.position.resize(11);
    pub_joint_cmd_msg.name = {"joint1", "joint2", "joint3", "joint4",
                              "joint5", "joint6", "joint7", "joint8",
                              "joint9", "joint10", "joint11"};
  }

  VectorXd q_sol = q_init;

  auto start_cb_time = std::chrono::high_resolution_clock::now();
  auto start_it_time = std::chrono::high_resolution_clock::now();
  auto stop_it_time = std::chrono::high_resolution_clock::now();

  for (uint i = 0; i < num_samples; i++)
  {
    int res = 1;
    pin::SE3 x_des = path[i];

    if (print_all)
      std::cout << "\n----------------------------\n[Prob " << i
                << "] Solving for Pos: " << path[i].translation().transpose()
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

    if (!res)
    {
      VectorXd::Map(&pub_joint_cmd_msg.position[0],
                    pub_joint_cmd_msg.position.size()) = q_sol;
      pub_joint_sol.publish(pub_joint_cmd_msg);

      success_time_it += duration_it.count();
      n_success++;
      q_init = q_sol;
    }

    total_time_it += duration_it.count();
    usleep(100000);
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

  return;
}

bool kill_process = false;
void SigIntHandler(int signal)
{
  kill_process = true;
  ROS_INFO_STREAM("SHUTDOWN SIGNAL RECEIVED");
}

std::vector<pin::SE3> createSquarePath(Vector3d center)
{
  int n_points_ = 4000;
  Vector3d point_tr;
  Matrix3d point_rot;
  std::vector<pin::SE3> path_points;

  pin::SE3 point;

  // Path points for  EE w.r.t. Base Frame
  point_tr = center + Vector3d(-0.05, -0.05, 0.0);
  point_rot << 0, 1, 0, -1, 0, 0, 0, 0, 1; // Looking forward
  path_points.clear();

  for (int i = 0; i < n_points_; i++)
  {
    if (i < 1000)
    {
      point_tr = point_tr + Vector3d(0.0001, 0, 0);
    }
    else if (i < 2000 && i > 1000)
    {
      point_tr = point_tr + Vector3d(0.0, 0.0001, 0);
    }
    else if (i < 3000 && i > 2000)
    {
      point_tr = point_tr - Vector3d(0.0001, 0, 0);
    }
    else
    {
      point_tr = point_tr - Vector3d(0.0, 0.0001, 0);
    }
    point = pin::SE3(point_rot, point_tr);

    path_points.push_back(point);
  }
  return path_points;
}

std::vector<pin::SE3> create6DCircularPath(Vector3d center, int ori_idx,
                                           double radius, int n_points)
{
  Vector3d point_tr;
  Matrix3d point_rot;
  std::vector<pin::SE3> path_points;

  pin::SE3 point;

  // Path points for  EE w.r.t. Base Frame
  point_tr = center;
  if (ori_idx == 1)
    point_rot << 0, 0, 1, 1, 0, 0, 0, 1, 0; // Pointing down (R2)
  else if (ori_idx == 2)
    point_rot << -1, 0, 0, 0, 0, 1, 0, 1, 0; // Pointing down (R3)
  else if (ori_idx == 3)
    point_rot << 1, 0, 0, 0, 0, 1, 0, -1, 0; // Pointing up (R3)
  else
    point_rot << 0, 0, 1, 1, 0, 0, 0, 1, 0; // Pointing down (R2)

  path_points.clear();
  for (int i = 0; i < n_points; i++)
  {
    point_tr =
        center +
        radius * Vector3d(cos((static_cast<double>(i) / n_points) * 2 * M_PI),
                          sin((static_cast<double>(i) / n_points) * 2 * M_PI),
                          0);
    point = pin::SE3(point_rot, point_tr);
    path_points.push_back(point);
  }

  return path_points;
}

std::vector<pin::SE3> create6DHelixPath(Vector3d center, int ori_idx,
                                        double radius, int n_points)
{
  Vector3d point_tr;
  Matrix3d point_rot;
  std::vector<pin::SE3> path_points;

  pin::SE3 point;

  // Path points for  EE w.r.t. Base Frame
  point_tr = center;
  if (ori_idx == 1)
    point_rot << 0, 0, 1, 1, 0, 0, 0, 1, 0; // Pointing down (R2)
  else if (ori_idx == 2)
    point_rot << -1, 0, 0, 0, 0, 1, 0, 1, 0; // Pointing down (R3)
  else if (ori_idx == 3)
    point_rot << 1, 0, 0, 0, 0, 1, 0, -1, 0; // Pointing up (R3)
  else
    point_rot << 0, 0, 1, 1, 0, 0, 0, 1, 0; // Pointing down (R2)

  path_points.clear();
  for (int i = 0; i < n_points; i++)
  {
    double st = 4 * static_cast<double>(i) / n_points;

    point_tr = center +
               radius * Vector3d(cos(st * 2 * M_PI), sin(st * 2 * M_PI), 0) +
               Vector3d(0, 0, st * 0.01);
    point = pin::SE3(point_rot, point_tr);
    path_points.push_back(point);
  }

  return path_points;
}

std::vector<pin::SE3> create3DCircularPath(Vector3d center, Vector3d rcm,
                                           double radius, int n_points)
{
  Vector3d point_tr;
  Matrix3d point_rot;
  std::vector<pin::SE3> path_points;

  pin::SE3 point;

  // Path points for  EE w.r.t. Base Frame
  point_tr = center;
  point_rot << -1, 0, 0, 0, 0, 1, 0, 1, 0; // Pointing down

  path_points.clear();
  for (int i = 0; i < n_points; i++)
  {
    point_tr =
        center +
        radius * Vector3d(cos((static_cast<double>(i) / n_points) * 2 * M_PI),
                          sin((static_cast<double>(i) / n_points) * 2 * M_PI),
                          0);
    Vector3d vz = point_tr - rcm;
    Vector3d uz = vz / vz.norm();

    Vector3d uy = uz.cross(Vector3d(0, 1, 0));
    Vector3d ux = uy.cross(uz);
    point_rot.col(0) = ux;
    point_rot.col(1) = uy;
    point_rot.col(2) = uz;

    point = pin::SE3(point_rot, point_tr);
    path_points.push_back(point);
  }

  return path_points;
}

std::vector<pin::SE3> create3DHelixPath(Vector3d center, Vector3d rcm,
                                        double radius, int n_points)
{
  Vector3d point_tr;
  Matrix3d point_rot;
  std::vector<pin::SE3> path_points;

  pin::SE3 point;

  // Path points for EE w.r.t. Base Frame
  point_tr = center;

  point_rot << 0, 0, 1, 1, 0, 0, 0, 1, 0; // Pointing down (R2)

  path_points.clear();
  for (int i = 0; i < n_points; i++)
  {
    double st = 4 * static_cast<double>(i) / n_points;

    point_tr = center +
               radius * Vector3d(cos(st * 2 * M_PI), sin(st * 2 * M_PI), 0) +
               Vector3d(0, 0, st * 0.01);

    Vector3d vz = point_tr - rcm;
    Vector3d uz = vz / vz.norm();

    Vector3d uy = uz.cross(Vector3d(0, 1, 0));
    Vector3d ux = uy.cross(uz);
    point_rot.col(0) = ux;
    point_rot.col(1) = uy;
    point_rot.col(2) = uz;

    point = pin::SE3(point_rot, point_tr);
    path_points.push_back(point);
  }

  return path_points;
}

std::vector<pin::SE3> create6DLissajousPath(Vector3d init_pos,
                                            Vector3d coefficients, int ori_idx,
                                            int n_points)
{
  Vector3d point_tr;
  Matrix3d point_rot;
  std::vector<pin::SE3> path_points;

  double a_coef = coefficients[0];
  double b_coef = coefficients[1];
  double c_coef = coefficients[2];

  pin::SE3 point;

  // Path points for EE w.r.t. Base Frame

  if (ori_idx == 1)
    point_rot << 0, 0, 1, 1, 0, 0, 0, 1, 0; // Pointing down (R2)
  else if (ori_idx == 2)
    point_rot << -1, 0, 0, 0, 0, 1, 0, 1, 0; // Pointing down (R3)
  else if (ori_idx == 3)
    point_rot << 1, 0, 0, 0, 0, 1, 0, -1, 0; // Pointing up (R3)
  else
    point_rot << 0, 0, 1, 1, 0, 0, 0, 1, 0; // Pointing down (R2)

  path_points.clear();
  for (int i = 0; i < n_points; i++)
  {
    double st = 10 * static_cast<double>(i) / n_points;
    point_tr =
        init_pos + Vector3d(a_coef * sin(st), b_coef * sin(2 * (st + M_PI_2)),
                            c_coef * cos(2 * st) - c_coef);
    point = pin::SE3(point_rot, point_tr);
    path_points.push_back(point);
  }

  return path_points;
}

int main(int argc, char **argv)
{
  ROS_INFO("RCM constrained IK Benchmarking for COIKS");
  ros::init(argc, argv, "coiks_benchmark");
  ros::NodeHandle nh;
  std::signal(SIGINT, SigIntHandler);

  // ROS parameters
  std::string ik_solver;
  int max_iter;
  int n_points;
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

  if (!nh.getParam("n_points", n_points))
  {
    n_points = 100;
  }
  std::string path_type;
  if (!nh.getParam("path_type", path_type))
  {
    path_type = "6d";
  }
  double path_param_1;
  if (!nh.getParam("path_param_1", path_param_1))
  {
    path_param_1 = 0.005;
  }
  int path_param_2;
  if (!nh.getParam("path_param_2", path_param_2))
  {
    path_param_2 = 0;
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

  // Setting up URDF path
  std::string pkg_path = ros::package::getPath("coiks");
  std::string urdf_path;

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

  so.verb_level_ = verb_level;
  so.time_stats_ = time_stats;
  so.logging_enabled_ = logging_enabled;
  so.log_path_ = log_path;

  Vector3d path_center;

  switch (robot_id)
  {
  case 0:
    urdf_path = pkg_path + std::string("/urdf/") + "robot1_endoscope.urdf";
    so.prercm_joint_name_ = "joint_endoscope";
    so.postrcm_joint_name_ = "joint_tip";
    path_center << 0.506, 0.0, 0.370;
    so.trocar_pos_[0] = 0.426;
    so.trocar_pos_[1] = 0.0;
    so.trocar_pos_[2] = 0.485;
    break;

  case 1:
    urdf_path = pkg_path + std::string("/urdf/") + "robot2_openrst.urdf";
    so.prercm_joint_name_ = "joint_interface";
    so.postrcm_joint_name_ = "joint_pitch";
    path_center << 0.468, -0.024, 0.279;
    so.trocar_pos_[0] = 0.406;
    so.trocar_pos_[1] = -0.024;
    so.trocar_pos_[2] = 0.415;
    break;

  case 2:
    urdf_path = pkg_path + std::string("/urdf/") + "robot3_hyperrst.urdf";
    so.prercm_joint_name_ = "joint_interface";
    so.postrcm_joint_name_ = "joint_pitch_1";
    path_center << 0.476, -0.025, 0.264;
    so.trocar_pos_[0] = 0.406;
    so.trocar_pos_[1] = -0.024;
    so.trocar_pos_[2] = 0.415;
    break;

  default:
    ROS_ERROR("Kinematic Tree ID not recognized.");
    return -1;
  }
  ROS_INFO_STREAM("Using URDF found in: " << urdf_path);

  // Initialiing COIKS
  ROS_INFO("Starting CODCS-IK");
  COIKS coiks(urdf_path, "base_link", "link_ee", ik_solver, so, max_time,
              max_error, max_iter, dt);

  ROS_WARN("Running path following configurations for COIKS");
  std::vector<pin::SE3> path;

  if (path_type == "c6d")
    path =
        create6DCircularPath(path_center, path_param_2, path_param_1, n_points);
  else if (path_type == "c3d")
    path = create3DCircularPath(path_center, so.trocar_pos_, path_param_1,
                                n_points);
  else if (path_type == "l6d")
    path = create6DLissajousPath(path_center, Vector3d(0.04, 0.04, 0.02),
                                 path_param_1, n_points);
  else if (path_type == "h6d")
    path = create6DHelixPath(path_center, path_param_2, path_param_1, n_points);

  solveIkFromGivenList(nh, urdf_path, &coiks, "link_ee", path, robot_id,
                       (print_all & 4) >> 2);
  ROS_INFO("Benchmark finished");

  return 0;
}
