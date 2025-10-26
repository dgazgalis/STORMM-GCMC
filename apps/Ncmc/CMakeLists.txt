# NCMC GCMC test application
cmake_minimum_required(VERSION 3.12)

# Application name
set(APP_NAME "ncmc_test.${STORMM_APP_SUFFIX}")

# Source files
set(APP_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gcmc_ncmc_test_runner.cpp
)

# Create executable
add_executable(${APP_NAME} ${APP_SOURCES})

# Link against STORMM library
target_link_libraries(${APP_NAME} ${PROJECT_NAME})

# Installation
install(TARGETS ${APP_NAME}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
