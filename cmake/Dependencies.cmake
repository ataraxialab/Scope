set(Scope_LINKER_LIBS)

shadow_find_os_arch(Scope_Platform Scope_Arch)

set(Scope_INSTALL_INCLUDE_PREFIX include)
set(Scope_INSTALL_LIB_PREFIX lib/${Scope_Platform}/${Scope_Arch})
set(Scope_INSTALL_BIN_PREFIX bin)

set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${Scope_INSTALL_LIB_PREFIX})

find_package(tron)
if (tron_FOUND)
  include_directories(SYSTEM ${tron_INCLUDE_DIRS})
  list(APPEND Scope_LINKER_LIBS ${tron_LIBRARIES})
  install(FILES ${tron_LIBRARIES} DESTINATION ${Scope_INSTALL_LIB_PREFIX})
else()
  message(FATAL_ERROR "Could not find tron")
endif()

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

if (${USE_OpenCV})
  find_package(OpenCV PATHS ${OpenCV_DIR} NO_DEFAULT_PATH QUIET COMPONENTS core highgui imgproc imgcodecs videoio)
  if (NOT OpenCV_FOUND) # if not OpenCV 3.x, then try to find OpenCV 2.x in default path
    find_package(OpenCV REQUIRED QUIET COMPONENTS core highgui imgproc)
  endif ()
  if (${OpenCV_VERSION} VERSION_GREATER "2.4.13")
    find_package(OpenCV REQUIRED QUIET COMPONENTS core highgui imgproc imgcodecs videoio)
  endif ()
  include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
  list(APPEND Scope_LINKER_LIBS ${OpenCV_LIBS})
  message(STATUS "Found OpenCV: ${OpenCV_CONFIG_PATH} (found version ${OpenCV_VERSION})")
  add_definitions(-DUSE_OpenCV)
endif ()
