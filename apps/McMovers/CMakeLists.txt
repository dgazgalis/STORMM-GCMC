# MC Movers Test Application
# Tests translation, rotation, and torsion Monte Carlo moves

# Application name
set(APP_NAME "mc_movers_test.${STORMM_APP_SUFFIX}")

# Source files
set(APP_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mc_movers_test_runner.cpp
)

# Create executable
add_executable(${APP_NAME} ${APP_SOURCES})

# Link against STORMM library
target_link_libraries(${APP_NAME} ${PROJECT_NAME})

# Install
install(TARGETS ${APP_NAME} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
