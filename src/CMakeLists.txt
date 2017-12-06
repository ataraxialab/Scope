set(scope_src)
set(test_src)

add_subdirectory(test)
add_subdirectory(ot)
add_subdirectory(common)
add_subdirectory(Scope)

include_directories("common")


foreach (test_src ${test_src})
    get_filename_component(test_example_fil_we ${test_src} NAME_WE)
    add_executable(${test_example_fil_we} ${test_src} ${scope_src})
    target_link_libraries(${test_example_fil_we} ${Scope_LINKER_LIBS})
    install(TARGETS ${test_example_fil_we} DESTINATION ${Scope_INSTALL_BIN_PREFIX})
endforeach ()
