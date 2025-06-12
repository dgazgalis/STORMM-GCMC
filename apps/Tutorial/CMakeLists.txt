set(APP_NAME "tutorial_i.${STORMM_APP_SUFFIX}")
add_executable(${APP_NAME}
        ${CMAKE_CURRENT_SOURCE_DIR}/tutorial_i.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/hpc_tutorial_i.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/hpc_tutorial_i.h)
target_sources(${APP_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tutorial_i.cpp)
target_link_libraries(${APP_NAME} ${PROJECT_NAME})
install(TARGETS ${APP_NAME}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

set(APP_NAME "tutorial_ii.${STORMM_APP_SUFFIX}")
add_executable(${APP_NAME}
        ${CMAKE_CURRENT_SOURCE_DIR}/tutorial_ii.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/randomwalk.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/randomwalk.h
        ${CMAKE_CURRENT_SOURCE_DIR}/hpc_randomwalk.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/hpc_randomwalk.h)
target_sources(${APP_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tutorial_ii.cpp)
target_link_libraries(${APP_NAME} ${PROJECT_NAME})
install(TARGETS ${APP_NAME}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

set(APP_NAME "tutorial_iii.${STORMM_APP_SUFFIX}")
add_executable(${APP_NAME}
        ${CMAKE_CURRENT_SOURCE_DIR}/tutorial_iii.cpp)
target_sources(${APP_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tutorial_iii.cpp)
target_link_libraries(${APP_NAME} ${PROJECT_NAME})
install(TARGETS ${APP_NAME}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

set(APP_NAME "tutorial_iv.${STORMM_APP_SUFFIX}")
add_executable(${APP_NAME}
        ${CMAKE_CURRENT_SOURCE_DIR}/tutorial_iv.cpp)
target_sources(${APP_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tutorial_iv.cpp)
target_link_libraries(${APP_NAME} ${PROJECT_NAME})
install(TARGETS ${APP_NAME}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
