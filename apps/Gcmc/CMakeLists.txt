set(GCMC_APP "gcmc.${STORMM_APP_SUFFIX}")
add_executable(${GCMC_APP}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gcmc_runner.cpp)
target_link_libraries(${GCMC_APP} ${PROJECT_NAME})
install(TARGETS ${GCMC_APP}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

set(GCMC_TEST_APP "gcmc_test.${STORMM_APP_SUFFIX}")
add_executable(${GCMC_TEST_APP}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gcmc_test_runner.cpp)
target_link_libraries(${GCMC_TEST_APP} ${PROJECT_NAME})
install(TARGETS ${GCMC_TEST_APP}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
