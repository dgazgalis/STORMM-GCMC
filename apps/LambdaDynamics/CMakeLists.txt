# Lambda Dynamics Runner Application
# Enables parity testing with standard MD using lambda-scaled forces

set(APP_NAME "lambda_dynamics.${STORMM_APP_SUFFIX}")

# Create the executable with the main source file
add_executable(${APP_NAME}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lambda_dynamics_runner.cpp)

# Link against the main STORMM library
target_link_libraries(${APP_NAME} ${PROJECT_NAME})

# Install the executable to the bin directory
install(TARGETS ${APP_NAME}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# Print build status
message(STATUS "Configured Lambda Dynamics app: ${APP_NAME}")