set(Scope_LINKER_LIBS)

#shadow_find_os_arch(Scope_Platform Scope_Arch)

set(Scope_INSTALL_INCLUDE_PREFIX include)
set(Scope_INSTALL_LIB_PREFIX lib/${Scope_Platform}/${Scope_Arch})
set(Scope_INSTALL_BIN_PREFIX bin)

set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${Scope_INSTALL_LIB_PREFIX})

if (${USE_MONGO_CXX_DRIVER})
  find_package(libmongocxx REQUIRED QUIET)
  find_package(libbsoncxx REQUIRED)
  if (${libmongocxx_FOUND})
    include_directories(SYSTEM ${LIBMONGOCXX_INCLUDE_DIRS})
    include_directories(SYSTEM ${LIBBSONCXX_INCLUDE_DIRS})

    list(APPEND Scope_LINKER_LIBS ${LIBMONGOCXX_LIBRARIES})
    list(APPEND Scope_LINKER_LIBS ${LIBBSONCXX_LIBRARIES})
#    message(STATUS "Found CURL: ${CURL_LIBRARIES} (found version ${CURL_VERSION_STRING})")
    add_definitions(-DMONGO_CXX)
  else ()
    message(FATAL_ERROR "Could not find MONGO_CXX")
  endif ()
endif ()
