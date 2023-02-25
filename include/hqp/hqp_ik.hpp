#ifndef HQPIK_HPP
#define HQPIK_HPP

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/autodiff/casadi.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/spatial/explog.hpp>

// HQP Interfaces
#include <hqp/interfaces/hqp_casadi.hpp>

using namespace Eigen;
namespace pin = pinocchio;

namespace coiks
{

  class COIKS;

  template <typename T>
  class HQP_IkSolver
  {
    friend class coiks::COIKS;

  public:
    HQP_IkSolver(const pin::Model &_model, const pin::FrameIndex &_Fid,
                 SolverOptions _solver_opts, const double _max_time,
                 const double _max_error, const int _max_iter = 1e4,
                 const double _dt = 1)
    {
      hqp_solver_.reset(new T(_model, _Fid, _solver_opts, _max_time, _max_error,
                              _max_iter, _dt));
    }
    ~HQP_IkSolver(){};

    int IkSolve(const VectorXd q_init, const pin::SE3 &x_des, VectorXd &q_sol)
    {
      int res;
      res = hqp_solver_->IkSolve(q_init, x_des, q_sol);
      return res;
    }

    void abort();
    void reset();
    void set_max_time(double _max_time);
    void set_max_error(double _max_error);

  private:
    std::unique_ptr<T> hqp_solver_;
  };

  template <typename T>
  inline void HQP_IkSolver<T>::abort()
  {
    hqp_solver_->abort();
  }
  template <typename T>
  inline void HQP_IkSolver<T>::reset()
  {
    hqp_solver_->reset();
  }
  template <typename T>
  inline void HQP_IkSolver<T>::set_max_time(double _max_time)
  {
    hqp_solver_->set_max_time(_max_time);
  }
  template <typename T>
  inline void HQP_IkSolver<T>::set_max_error(double _max_error)
  {
    hqp_solver_->set_max_error(_max_error);
  }

} // namespace coiks

#endif
