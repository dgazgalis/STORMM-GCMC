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

set(GCMC_HYBRID_APP "gcmc_hybrid.${STORMM_APP_SUFFIX}")
add_executable(${GCMC_HYBRID_APP}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gcmc_hybrid_runner.cpp)
target_link_libraries(${GCMC_HYBRID_APP} ${PROJECT_NAME})
install(TARGETS ${GCMC_HYBRID_APP}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if (BUILD_TESTING AND STORMM_ENABLE_CUDA)
    set(GCMC_TEST_INPUT_DIR ${PROJECT_SOURCE_DIR}/apps/Gcmc/test_inputs)
    set(GCMC_TEST_OUTPUT_DIR ${PROJECT_BINARY_DIR}/gcmc_hybrid_regression)

    add_test(NAME gcmc_hybrid_constant
            COMMAND $<TARGET_FILE:${GCMC_HYBRID_APP}>
                    --config ${GCMC_TEST_INPUT_DIR}/hybrid_constant.cfg
                    --output-dir ${GCMC_TEST_OUTPUT_DIR}/constant)
    set_tests_properties(gcmc_hybrid_constant PROPERTIES
            ENVIRONMENT "STORMM_LOG_MEMORY=0"
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/apps/Gcmc/test_inputs)

    add_test(NAME gcmc_hybrid_adaptive
            COMMAND $<TARGET_FILE:${GCMC_HYBRID_APP}>
                    --config ${GCMC_TEST_INPUT_DIR}/hybrid_adaptive.json
                    --output-dir ${GCMC_TEST_OUTPUT_DIR}/adaptive)
    set_tests_properties(gcmc_hybrid_adaptive PROPERTIES
            ENVIRONMENT "STORMM_LOG_MEMORY=0"
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/apps/Gcmc/test_inputs)
endif()
