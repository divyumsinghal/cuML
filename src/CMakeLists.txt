# Create the LinearRegression library
add_library(LinearRegression LinearRegression.cu)

# Set CUDA properties
set_target_properties(LinearRegression PROPERTIES
    CUDA_ARCHITECTURES 89
    POSITION_INDEPENDENT_CODE ON
)

# Link with CUDA libraries and utils
target_link_libraries(LinearRegression
    utils
    CUDA::cublas
    CUDA::cuda_driver
)

# Include directories
target_include_directories(LinearRegression PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../
)