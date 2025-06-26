cmake_minimum_required(VERSION 3.21)

# 设置vcpkg工具链
if(DEFINED ENV{VCPKG_ROOT})
    message(STATUS "VCPKG_ROOT: $ENV{VCPKG_ROOT}")
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
else()
    message(FATAL_ERROR "VCPKG_ROOT is not set")
endif()

project(gpt2 VERSION 1.0.0 LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(fmt CONFIG REQUIRED)

# 设置编译选项
if (MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()





# 设置目标名称(ProjectName-SystemName)
set(TARGET ${PROJECT_NAME}-${CMAKE_SYSTEM_NAME})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR})



add_subdirectory(src)
