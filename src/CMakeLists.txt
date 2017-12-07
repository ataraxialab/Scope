set(scope_src)
set(test_src)

add_subdirectory(test)
add_subdirectory(ot)
add_subdirectory(common)
add_subdirectory(Scope)

include_directories("common")

add_library(scope SHARED ${scope_src})
target_link_libraries(scope ${Scope_LINKER_LIBS})
install(TARGETS scope DESTINATION ${Scope_INSTALL_LIB_PREFIX})
install(FILES Scope/atlab_scope.h DESTINATION ${Scope_INSTALL_INCLUDE_PREFIX})

foreach (test_src ${test_src})
    get_filename_component(test_example_fil_we ${test_src} NAME_WE)
    add_executable(${test_example_fil_we} ${test_src})
    target_link_libraries(${test_example_fil_we} scope)
    install(TARGETS ${test_example_fil_we} DESTINATION ${Scope_INSTALL_BIN_PREFIX})
endforeach ()
