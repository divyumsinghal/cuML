# Create the example executable
add_executable(MatrixOperationsExample MatrixOperationsExample.cpp)
add_executable(LinearRegressionExample LinearRegressionExample.cpp)

# Set C++ properties
set_target_properties(MatrixOperationsExample  PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
set_target_properties(LinearRegressionExample  PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Link with our libraries
target_link_libraries(MatrixOperationsExample
    MatrixOperations
    utils
    CUDA::cudart
)
target_link_libraries(LinearRegressionExample
    LinearRegression
    utils
    CUDA::cudart
)

# Include directories
target_include_directories(MatrixOperationsExample PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../
)
target_include_directories(LinearRegressionExample PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../
)