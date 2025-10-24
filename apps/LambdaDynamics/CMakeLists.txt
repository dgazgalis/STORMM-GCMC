# Lambda Dynamics Application
# Parity testing application matching dynamics.stormm.cuda structure
# Uses lambda-aware dynamics functions with all lambda=1.0 for verification

set(APP_NAME "lambda_dynamics.${STORMM_APP_SUFFIX}")

add_executable(${APP_NAME}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/setup.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/setup.h)

target_sources(${APP_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/simulator.cpp)

target_link_libraries(${APP_NAME} ${PROJECT_NAME})

install(TARGETS ${APP_NAME}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# Lambda Test Runner - Simple CLI tool for lambda scaling tests
# Accepts topology, coordinates, n_steps, and lambda value as command-line arguments
set(TEST_APP_NAME "lambda_test_runner.${STORMM_APP_SUFFIX}")

add_executable(${TEST_APP_NAME}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/lambda_dynamics_runner.cpp)

target_link_libraries(${TEST_APP_NAME} ${PROJECT_NAME})

install(TARGETS ${TEST_APP_NAME}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})