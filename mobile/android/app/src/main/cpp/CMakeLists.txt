cmake_minimum_required(VERSION 3.22.1)
project(elrslang_smoke)

add_library(elrslang_smoke SHARED native-lib.cpp)

find_library(log-lib log)
find_library(android-lib android)
find_library(vulkan-lib vulkan)

target_link_libraries(elrslang_smoke ${log-lib} ${android-lib} ${vulkan-lib})
