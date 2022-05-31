cmake_minimum_required(VERSION 3.16.2) # For boost 1.72

project(MultiView VERSION 0.1.0 DESCRIPTION "MultiView")

if("${MULTIVIEW_DEPS_DIR}" STREQUAL "")
  message(FATAL_ERROR "Set MULTIVIEW_DEPS_DIR as the path to a conda env having the dependencies.")
endif()

set(CMAKE_BUILD_TYPE "Release")
message(STATUS "Setting build type: ${CMAKE_BUILD_TYPE}")

# Where to search for dependencies
set(CMAKE_PREFIX_PATH "${MULTIVIEW_DEPS_DIR};${CMAKE_PREFIX_PATH}")
set(CMAKE_MODULE_PATH "${MULTIVIEW_DEPS_DIR}/lib/cmake/Ceres;${CMAKE_MODULE_PATH}")

set(CMAKE_CXX_STANDARD 11)

# Image libs
find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)
find_package(TIFF REQUIRED)

# GFlags 
find_package(Gflags REQUIRED)
if (GFLAGS_FOUND)
   message("-- Found Google Flags: ${GFLAGS_INCLUDE_DIRS} in namespace: ${GFLAGS_NAMESPACE}")
else (GFLAGS_FOUND)
   message(FATAL_ERROR "Can't find Google FLAGS.")
endif (GFLAGS_FOUND)

# GLog
find_package(Glog REQUIRED)
if (GLOG_FOUND)
  message("-- Found Google Logging: ${GLOG_INCLUDE_DIRS}")
else (GLOG_FOUND)
  message(FATAL_ERROR "Can't find Glog. Please set GLOG_INCLUDE_DIR & "
    "GLOG_LIBRARY")
endif (GLOG_FOUND)

# Eigen
find_package(Eigen3 3 REQUIRED NO_MODULE)
if (EIGEN3_FOUND)
   message("-- Found Eigen3: ${EIGEN3_INCLUDE_DIRS}")
else (EIGEN3_FOUND)
   message(FATAL_ERROR "Cannot find Eigen3.")
endif (EIGEN3_FOUND)
include_directories("${CMAKE_CURRENT_SOURCE_DIR};${EIGEN3_INCLUDE_DIR}")

# Ceres
find_package(Ceres REQUIRED SuiteSparse)
if (CERES_FOUND)
   message("-- Found Ceres: ${CERES_INCLUDE_DIRS}")
else (CERES_FOUND)
  message(FATAL_ERROR "Can't find Ceres. Please set CERES_INCLUDE_DIR and "
    "CERES_LIBRARY")
endif (CERES_FOUND)

# Boost
set(Boost_USE_RELEASE_LIBS   ON)
set(Boost_USE_MULTITHREADED  ON)
set(BUILD_SHARED_LIBS        ON)
set(Boost_USE_STATIC_RUNTIME OFF)
set(Boost_USE_STATIC_LIBS OFF)
set(BOOST_ROOT       "${MULTIVIEW_DEPS_DIR}")
set(BOOST_LIBRARYDIR "${MULTIVIEW_DEPS_DIR}/lib")
set(BOOST_INCLUDEDIR "${MULTIVIEW_DEPS_DIR}/include")
set(Boost_NO_SYSTEM_PATHS ON) # Do not search anywhere else
set(Boost_DEBUG ON) # to help figure out issues
set(Boost_DETAILED_FAILURE_MSG ON)
find_package(Boost 1.70 REQUIRED COMPONENTS filesystem)
if (Boost_FOUND)
    message("Successfully found Boost version ${Boost_VERSION}.")
    message("Boost include dir = ${Boost_INCLUDE_DIRS}")
    message("Boost library dir = ${Boost_LIBRARY_DIRS}")
    message("Boost libraries   = ${Boost_LIBRARIES}")
    # TODO(oalexan1): Not sure about the lines below
    include_directories(${Boost_INCLUDE_DIRS})
    link_directories(${Boost_LIBRARY_DIRS})
else (Boost_FOUND)
    message(FATAL_ERROR "Can't find boost.")
endif (Boost_FOUND)

# OpenCV
find_package(OpenCV 3.2.0 REQUIRED)
# Find OpenCV and fix a 3.3.1 bug
find_package(OpenCV 3 REQUIRED)
if (${OpenCV_VERSION} MATCHES "3.3.1")
  foreach(__cvcomponent ${OpenCV_LIB_COMPONENTS})
    set (__original_cvcomponent ${__cvcomponent})
    if(NOT __cvcomponent MATCHES "^opencv_")
      set(__cvcomponent opencv_${__cvcomponent})
    endif()
    if (TARGET ${__cvcomponent})
      set_target_properties(${__cvcomponent} PROPERTIES
          MAP_IMPORTED_CONFIG_DEBUG ""
          MAP_IMPORTED_CONFIG_RELEASE ""
          MAP_IMPORTED_CONFIG_RELWITHDEBINFO ""
          MAP_IMPORTED_CONFIG_MINSIZEREL ""
      )
    endif()
  endforeach(__cvcomponent)
endif()
set(OpenCV_LIBRARIES ${OpenCV_LIBS})
if (OPENCV_FOUND)
   message("-- Found OpenCV: ${OPENCV_INCLUDE_DIRS}")
else (OPENCV_FOUND)
   message(FATAL_ERROR "Cannot find OpenCV.")
endif (OPENCV_FOUND)

# OpenMP
FIND_PACKAGE(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Build texrecon (the repo name is mvs-texturing)
add_subdirectory(texrecon)

# TODO(oalexan1): The code needs to be compiled a few times till it succeeds
# This should be fixed by making mve, mapmap, and rayint submodules.
# But note that rayint builds its own copy of mve which complicates things.

# Set up the include files for texrecon
set(TEXRECON_INCLUDE_DIRS 
    ${CMAKE_CURRENT_SOURCE_DIR}/elibs/rayint/libs
    ${CMAKE_CURRENT_SOURCE_DIR}/elibs/mve/libs
    ${CMAKE_CURRENT_SOURCE_DIR}/texrecon/libs)
include_directories(${TEXRECON_INCLUDE_DIRS})

# Set up the texrecon libraries
set(TEXRECON_DIR ${CMAKE_BINARY_DIR}/texrecon)
set(TEXRECON_LIBRARIES
    ${TEXRECON_DIR}/libs/tex/libtex.a
    # TODO(oalexan1): Link these dynamically and add as submodules
    # Then move them from the source directory to the build directory
    ${CMAKE_CURRENT_SOURCE_DIR}/elibs/mve/libs/mve/libmve.a
    ${CMAKE_CURRENT_SOURCE_DIR}/elibs/mve/libs/util/libmve_util.a)

add_subdirectory(camera_model)
add_subdirectory(openMVG)
add_subdirectory(rig_calibrator)

add_executable(rig_calibrator_tool tools/rig_calibrator.cc)
target_link_libraries(rig_calibrator_tool rig_camera_model rig_calibrator rig_openMVG
    ${TEXRECON_LIBRARIES} ${OpenCV_LIBRARIES}
    ${CERES_LIBRARIES} ${JPEG_LIBRARIES} ${PNG_LIBRARIES} ${TIFF_LIBRARIES}
    gflags glog Boost::filesystem)
