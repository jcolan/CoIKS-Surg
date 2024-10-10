# CoIKS-Surg: Concurrent Inverse Kinematics Solver for Surgical Robotic Manipulators
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

## Publications
This code has been implemented for the following publication. For further references please follow the link:

* Colan, Jacinto, et al. "[A Concurrent Framework for Constrained Inverse Kinematics of Minimally Invasive Surgical Robots](https://www.mdpi.com/1424-8220/23/6/3328)." Sensors 23.6 (2023): 3328.

