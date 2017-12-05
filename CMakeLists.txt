cmake_minimum_required(VERSION 3.7)
project(Scope)

if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 11)
endif()

set(test_src)
option(USE_MONGO_CXX_DRIVER "Use mongo CXX drivers" ON)
include(cmake/Dependencies.cmake)

add_subdirectory(test)

foreach (test_src ${test_src})
    get_filename_component(tron_example_fil_we ${test_src} NAME_WE)
    add_executable(${tron_example_fil_we} ${test_src})
    target_link_libraries(${tron_example_fil_we} ${Scope_LINKER_LIBS})
    install(TARGETS ${tron_example_fil_we} DESTINATION ${Scope_INSTALL_BIN_PREFIX})
endforeach ()
