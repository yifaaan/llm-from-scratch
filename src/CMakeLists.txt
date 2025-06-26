add_executable(${TARGET} gpt2.cpp)
target_link_libraries(${TARGET} fmt::fmt)

# 按编译器分别加入 AVX2/FMA 选项
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${TARGET} PRIVATE -O3 -mavx2 -mfma)
elseif (MSVC)
    target_compile_options(${TARGET} PRIVATE /O2 /arch:AVX2)
endif()