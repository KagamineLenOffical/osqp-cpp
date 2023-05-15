# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.20)

project(osqp-cpp)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# Options.
option(OSQP-CPP_BUILD_TESTS
  "Whether to build tests." ON
)

include(FetchContent)
#set(Eigen3_DIR D:/linear_solver/vcpkg/packages/eigen3_x64-windows/share/eigen3)
#set(absl_DIR D:/linear_solver/vcpkg/packages/abseil_x64-windows/share/absl)
find_package(GTest CONFIG REQUIRED)
find_package(osqp REQUIRED)
find_package(gflags CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(absl CONFIG REQUIRED)
find_package(libigl CONFIG REQUIRED)
find_package(tinyxml2 CONFIG REQUIRED)
enable_language(CUDA)
find_package(CUDA 11.3 REQUIRED)
## ABSL
#if (NOT TARGET absl::strings OR NOT TARGET absl::status OR NOT TARGET absl::span)
#  message(STATUS "osqp-cpp: `absl` targets not found. Attempting to fetch contents...")
#  FetchContent_Declare(
#    abseil-cpp
#    GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
#    GIT_TAG        origin/master
#  )
#  FetchContent_MakeAvailable(abseil-cpp)
#else()
#  message(STATUS "osqp-cpp: `absl` targets found.")
#endif()
#
## EIGEN
## Eigen is usually available as a system package, so we use find_package.
#if (NOT TARGET Eigen3::Eigen)
#  message(STATUS "osqp-cpp: `Eigen3` targets not found. Attempting to find package...")
#  find_package(Eigen3 3.3.7 REQUIRED NO_MODULE)
#else()
#  message(STATUS "osqp-cpp: `Eigen3` targets found.")
#endif()
#
## OSQP
#if (NOT TARGET osqpstatic)
#  message(STATUS "osqp-cpp: `osqp` targets not found. Attempting to fetch contents...")
#  FetchContent_Declare(
#    osqp
#    GIT_REPOSITORY https://github.com/oxfordcontrol/osqp.git
#    GIT_TAG        origin/master
#  )
#  FetchContent_MakeAvailable(osqp)
#else()
#  message(STATUS "osqp-cpp: `osqp` targets found.")
#endif()
#
## Googletest
#if (OSQP-CPP_BUILD_TESTS)
#  enable_testing()
#  if (NOT TARGET gtest OR NOT TARGET gmock OR NOT TARGET gtest_main)
#    message(STATUS "osqp-cpp: `googletest` targets not found. Attempting to fetch contents...")
#    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
#    FetchContent_Declare(
#      googletest
#      GIT_REPOSITORY https://github.com/google/googletest.git
#      GIT_TAG        origin/main
#    )
#    FetchContent_MakeAvailable(googletest)
#    include(GoogleTest)
#  else()
#    message(STATUS "osqp-cpp: `googletest` targets found.")
#  endif()
#endif()

message(STATUS "osqp-cpp: Adding osqp-cpp library...")
add_library(osqp-cpp src/osqp++.cc)
target_link_libraries(osqp-cpp PUBLIC absl::strings absl::status absl::statusor Eigen3::Eigen PRIVATE osqp::osqpstatic ${CMAKE_DL_LIBS})
target_include_directories(osqp-cpp PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
message(STATUS "osqp-cpp: Added osqp-cpp library.")

# Build and register tests.
enable_testing()
include(GoogleTest)

#if (OSQP-CPP_BUILD_TESTS)
#  message(STATUS "osqp-cpp: Adding osqp-cpp tests...")
add_executable(osqp_test test/osqp++_test.cc)
target_link_libraries(osqp_test gtest gmock gtest_main absl::status absl::span osqp-cpp)
gtest_discover_tests(osqp_test)

add_executable(osqp_test_2 test/osqp++_test2.cpp)
target_link_libraries(osqp_test_2  absl::status absl::span osqp-cpp gflags::gflags igl::igl_core tinyxml2::tinyxml2)

#  message(STATUS "osqp-cpp: Added osqp-cpp tests.")
#endif()

