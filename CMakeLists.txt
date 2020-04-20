cmake_minimum_required(VERSION 3.15)
project(matrix_norm_omp C)

set(CMAKE_C_STANDARD 99)

add_executable(matrix_norm_omp matrix_norm_omp.c)

include_directories(
        /usr/local/opt/libomp/include
)

link_directories(
        /usr/local/opt/libomp/lib
)

target_link_libraries(matrix_norm_omp /usr/local/opt/libomp/lib/libomp.a)
