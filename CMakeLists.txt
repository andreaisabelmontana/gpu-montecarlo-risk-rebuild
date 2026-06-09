cmake_minimum_required(VERSION 3.18)
project(montecarlo_risk LANGUAGES C)

set(CMAKE_C_STANDARD 11)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_C_FLAGS_RELEASE "-O3 -ffast-math")

include_directories(include)

add_executable(mc_serial src/serial.c src/common.c)
target_link_libraries(mc_serial m)

find_package(OpenMP)
add_executable(mc_openmp src/openmp.c src/common.c)
if(OpenMP_C_FOUND)
  target_link_libraries(mc_openmp PUBLIC OpenMP::OpenMP_C m)
else()
  target_link_libraries(mc_openmp m)
endif()

# Optional GPU backend — enabled only when a CUDA toolkit is present.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  add_executable(mc_cuda src/cuda.cu src/common.c)
  set_target_properties(mc_cuda PROPERTIES CUDA_ARCHITECTURES "70;75;80;86")
  target_link_libraries(mc_cuda curand)
  message(STATUS "CUDA found — mc_cuda will be built.")
else()
  message(STATUS "CUDA not found — skipping mc_cuda (CPU backends still build).")
endif()
