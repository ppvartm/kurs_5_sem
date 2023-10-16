cmake_minimum_required(VERSION 3.5)

project(modeling_of_gauss_process)


add_executable(modeling_of_gauss_process modeling_of_gauss_process/main.cpp)

set_target_properties(
      modeling_of_gauss_process PROPERTIES
      CXX_STANDART 11
      CXX_STANDART_REQUIRED ON)