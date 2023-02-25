# CoIKS: Concurrent Inverse Kinematics Solvers
This package contains a benchmark of concurrent inverse kinematics solvers for robot-assisted minimally invasive surgical robots.
The benchmark include unconstrained and constrained scenarios, where the constraints are defined by a set of joint limits and a remote-center-of-motion (RCM).

Concurrent IK solvers are based on the concurrent deployment of the single-method IK solvers:
- INVJ: Inverse Jacobian IK solver
- NLO: Nonlinear Optimization IK solver
- HQP: Hierarchical Quadratic Programming IK solver

The following concurrent ik solvers are implemented and benchmarked:
- INVJ+NLO
- INVJ+HQP
- HQP+NLO
- INVJ+NLO+HQP

Prerequisities;
* TRAC-IK
* Orocos-KDL
* Pinocchio
* CASADI (with the plugins for IPOPT and OSQP)
* IPOPT
* OSQP


