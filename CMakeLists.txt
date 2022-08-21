project(multispline)
cmake_minimum_required(VERSION 3.23)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(ROOT 6.22 CONFIG REQUIRED)

add_compile_options(-Wall -Wextra -O3 -ffast-math -mavx -mavx2)
# add_compile_options(-O0 -g)

add_executable(multispline multispline.cxx)
target_link_libraries(multispline PUBLIC ROOT::Hist)

add_executable(make_random_inputs make_random_inputs.cxx)
target_link_libraries(make_random_inputs PUBLIC ROOT::Hist)
