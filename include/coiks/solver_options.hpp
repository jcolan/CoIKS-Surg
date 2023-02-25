#ifndef SOLVER_OPTIONS_HPP
#define SOLVER_OPTIONS_HPP

// Eigen
#include <Eigen/Dense>
// C++
#include <map>
#include <vector>

using namespace Eigen;

namespace coiks
{

  class SolverOptions
  {
public:
    std::string error_type_;
    bool        manipulability_enabled_;

    // RCM Constraint control variables
    Vector3d    trocar_pos_;
    bool        constrained_control_;
    std::string prercm_joint_name_;
    std::string postrcm_joint_name_;
    bool        rcm_is_cost_;
    double      rcm_error_max_;

    // INVJ Variables
    double invj_Ke1_;
    double invj_Ke2_;

    // NLO variables
    bool                nlo_concurrent_;
    int                 nlo_concurrent_iterations_;
    std::string         nlo_linear_solver_;
    std::string         nlo_error_type_;
    std::vector<double> cost_coeff_;
    std::string         nlo_warm_start_;

    // HQP variables
    double hqp_Kd1_;
    double hqp_Kd2_;
    double hqp_Ke1_;
    double hqp_Ke2_;
    bool   hqp_warm_start_;

    // Logging variables
    bool                               logging_enabled_;
    std::string                        log_path_;
    int                                verb_level_;
    std::string                        time_stats_;
    std::map<std::string, std::string> other_opts_;
  };
} // namespace coiks
#endif