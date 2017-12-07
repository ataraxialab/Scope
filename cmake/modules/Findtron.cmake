include(FindPackageHandleStandardArgs)

set(tron_ROOT_DIR ${PROJECT_SOURCE_DIR}/third_party/tron/build CACHE PATH "Folder contains tron")

set(tron_DIR ${tron_ROOT_DIR})
message("aa")
find_path(tron_INCLUDE_DIRS
        NAMES tron_algorithm.hpp
        PATHS ${tron_DIR}
        PATH_SUFFIXES include include/x86_64 include/x64
        DOC "tron include"
        NO_DEFAULT_PATH)

if (NOT MSVC)
    find_library(shadow_LIBRARIES
            NAMES shadow
            PATHS ${tron_DIR}
            PATH_SUFFIXES lib lib64 lib/x86_64 lib/x86_64-linux-gnu lib/x64 lib/x86 lib/darwin/x86_64
            DOC "tron library"
            NO_DEFAULT_PATH)
    find_library(tron_LIBRARIES
            NAMES tron
            PATHS ${tron_DIR}
            PATH_SUFFIXES lib lib64 lib/x86_64 lib/x86_64-linux-gnu lib/x64 lib/x86 lib/darwin/x86_64
            DOC "tron library"
            NO_DEFAULT_PATH)
    set(tron_LIBRARIES ${shadow_LIBRARIES} ${tron_LIBRARIES})
else ()
    find_library(tron_LIBRARIES_RELEASE
            NAMES tron
            PATHS ${tron_DIR}
            PATH_SUFFIXES lib lib64 lib/x86_64 lib/x86_64-linux-gnu lib/x64 lib/x86
            DOC "tron library"
            NO_DEFAULT_PATH)
    find_library(tron_LIBRARIES_DEBUG
            NAMES trond
            PATHS ${tron_DIR}
            PATH_SUFFIXES lib lib64 lib/x86_64 lib/x86_64-linux-gnu lib/x64 lib/x86
            DOC "tron library"
            NO_DEFAULT_PATH)

    find_library(proto_LIBRARIES_RELEASE
            NAMES proto
            PATHS ${tron_DIR}
            PATH_SUFFIXES lib lib64 lib/x86_64 lib/x86_64-linux-gnu lib/x64 lib/x86
            DOC "proto library"
            NO_DEFAULT_PATH)
    find_library(proto_LIBRARIES_DEBUG
            NAMES protod
            PATHS ${tron_DIR}
            PATH_SUFFIXES lib lib64 lib/x86_64 lib/x86_64-linux-gnu lib/x64 lib/x86
            DOC "proto library"
            NO_DEFAULT_PATH)
    set(tron_LIBRARIES optimized ${tron_LIBRARIES_RELEASE} optimized ${proto_LIBRARIES_RELEASE} debug ${tron_LIBRARIES_DEBUG} debug ${proto_LIBRARIES_DEBUG})
endif ()

find_package_handle_standard_args(tron DEFAULT_MSG tron_INCLUDE_DIRS tron_LIBRARIES)

if (tron_FOUND)
    if (NOT tron_FIND_QUIETLY)
        message(STATUS "Found tron: ${tron_INCLUDE_DIRS}, ${tron_LIBRARIES}")
    endif ()
    mark_as_advanced(tron_ROOT_DIR tron_INCLUDE_DIRS tron_LIBRARIES tron_LIBRARIES_RELEASE tron_LIBRARIES_DEBUG proto_LIBRARIES_RELEASE proto_LIBRARIES_DEBUG)
else ()
    if (tron_FIND_REQUIRED)
        message(FATAL_ERROR "Could not find tron")
    endif ()
endif ()
