# Create the MatrixOperations library
add_library(MatrixOperations MatrixOperations.cu)

# Set CUDA properties
set_target_properties(MatrixOperations PROPERTIES
    CUDA_ARCHITECTURES 89
    POSITION_INDEPENDENT_CODE ON
)

# Link with CUDA libraries and utils
target_link_libraries(MatrixOperations
    utils
    CUDA::cublas
    CUDA::cuda_driver
)

# Include directories
target_include_directories(MatrixOperations PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../
)