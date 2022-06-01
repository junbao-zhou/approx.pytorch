execute_process(
    COMMAND python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())"
    OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND python -c "import distutils.sysconfig as sysconfig; import os; print(os.path.join(sysconfig.get_config_var('LIBDIR'), sysconfig.get_config_var('LDLIBRARY')))"
    OUTPUT_VARIABLE PYTHON_LIBRARY
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(PythonLibs REQUIRED)

# set(CMAKE_PREFIX_PATH "/usr/local/lib/python3.6/dist-packages/torch/share/cmake")
set(CMAKE_PREFIX_PATH "/opt/conda/lib/python3.7/site-packages/torch/share/cmake/")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(pybind11 CONFIG)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(include)

add_executable(
    test_cuda
    test_cuda.cu
)

pybind11_add_module(pybind_example pybind_example.cpp)

message("${TORCH_LIBRARIES}")
set(TORCH_LIBRARIES "${TORCH_LIBRARIES};/opt/conda/lib/python3.7/site-packages/torch/lib/libtorch_python.so")

# add_library(
#     tensor_compute
#     SHARED tensor_compute.cu
# )
# target_link_libraries(
#     tensor_compute
#     PUBLIC "${TORCH_LIBRARIES}"
# )
# set_target_properties(
#     tensor_compute
#     PROPERTIES CUDA_SEPARABLE_COMPILATION ON
#     CXX_STANDARD 14
# )

add_library(
    util
    SHARED util.cpp
)

add_executable(
    test_tensor
    test_tensor.cu
)
target_link_libraries(
    test_tensor
    PUBLIC util "${TORCH_LIBRARIES}" "${PYTHON_LIBRARY}"
)
set_target_properties(
    test_tensor
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON
    CXX_STANDARD 14
)
