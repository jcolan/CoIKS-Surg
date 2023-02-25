cmake_minimum_required(VERSION 3.0.2)
project(coiks)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
    trac_ik_lib
    kdl_parser
    roscpp
    roslib
    eigen_conversions
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)

# KDL
find_package(orocos_kdl REQUIRED)

# Pinnochio
find_package(pinocchio REQUIRED)

# NLOPT
# find_package(PkgConfig REQUIRED)
# pkg_check_modules(pkg_nlopt REQUIRED nlopt)

# Casadi
list(APPEND CMAKE_PREFIX_PATH "/usr/local/")
find_package(CASADI REQUIRED)

# RELEASE TYPE
set(CMAKE_BUILD_TYPE Release)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
 INCLUDE_DIRS include
 LIBRARIES coiks
 CATKIN_DEPENDS roscpp 
 DEPENDS
 )

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${pkg_nlopt_INCLUDE_DIRS}
    # ${orocos_kdl_INCLUDE_DIRS}
)

## Declare a C++ library
add_library(coiks 
    src/coiks.cpp
)


# add_dependencies(coiks pinocchio::pinocchio Eigen3::Eigen  casadi)


target_link_libraries(coiks 
    Eigen3::Eigen 
    pinocchio::pinocchio 
    ${catkin_LIBRARIES}
    ${pkg_nlopt_LIBRARIES}
    casadi
)

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
add_executable(unconstrained_benchmark src/unconstrained_benchmark.cpp)
add_executable(constrained_benchmark src/constrained_benchmark.cpp)

target_link_libraries(unconstrained_benchmark 
    ${catkin_LIBRARIES}
    coiks
    # ${orocos_kdl_LIBRARIES}
)

target_link_libraries(constrained_benchmark 
    ${catkin_LIBRARIES}
    coiks
    # ${orocos_kdl_LIBRARIES}
)
