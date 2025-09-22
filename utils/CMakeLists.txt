# Utils headers-only library
add_library(utils INTERFACE)

# Include directories for the interface library
target_include_directories(utils INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../
)

# Link with CUDA libraries
target_link_libraries(utils INTERFACE
    CUDA::cublas
    CUDA::cusolver
    CUDA::cufft
    CUDA::cusparse
)
