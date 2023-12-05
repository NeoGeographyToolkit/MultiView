cmake_minimum_required(VERSION 3.15.5)

project(MultiView VERSION 0.1.0 DESCRIPTION "MultiView")

if("${MULTIVIEW_DEPS_DIR}" STREQUAL "")
  message(FATAL_ERROR "Set MULTIVIEW_DEPS_DIR as the path to a conda env having the dependencies.")
endif()

set(CMAKE_BUILD_TYPE "Release")
message(STATUS "Setting build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_STANDARD 11)

# Where to search for dependencies
set(CMAKE_PREFIX_PATH "${MULTIVIEW_DEPS_DIR};${CMAKE_PREFIX_PATH}")
set(CMAKE_MODULE_PATH "${MULTIVIEW_DEPS_DIR}/lib/cmake/Ceres;${CMAKE_MODULE_PATH}")

# Where to search for include files
include_directories("${MULTIVIEW_DEPS_DIR}/include")

set(CMAKE_INSTALL_LIBDIR "lib") # install in lib and not lib64

# OpenMP
# TODO(oalexan1): Figure out why clang does not find OpenMP
# and remove the clucky logic below.
if (APPLE)
    find_package(OpenMP)
    set(OPENMP_INCLUDES "${MULTIVIEW_DEPS_DIR}/include")
    set(OpenMP_C_LIB_NAMES "gomp")
    set(OpenMP_CXX_LIB_NAMES "gomp")
    set(OpenMP_C_FLAGS "-fopenmp")
    set(OpenMP_CXX_FLAGS "-fopenmp")
    set(OpenMP_gomp_LIBRARY "${MULTIVIEW_DEPS_DIR}/lib/libgomp${CMAKE_SHARED_LIBRARY_SUFFIX}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_gomp_LIBRARY}")
    set(OpenMP_CXX_FOUND ON)
    set(WITH_OPENMP ON)
    message(STATUS "Manually set: OpenMP_CXX_FLAGS=${OpenMP_CXX_FLAGS}")
    message(STATUS "Mamually set: OpenMP_CXX_LIB_NAMES=${OpenMP_CXX_FLAGS}")
else()
    find_package(OpenMP REQUIRED)
endif()
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

# Image libs
find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)
find_package(TIFF REQUIRED)

# TBB
FIND_PACKAGE(TBB REQUIRED)

# GFlags 
find_package(Gflags REQUIRED)
if (Gflags_FOUND)
   message("-- Found Google Flags: ${GFLAGS_INCLUDE_DIRS} in namespace: ${GFLAGS_NAMESPACE}")
else (Gflags_FOUND)
   message(FATAL_ERROR "Can't find Google FLAGS.")
endif (Gflags_FOUND)

# GLog
find_package(Glog REQUIRED)
if (Glog_FOUND)
  message("-- Found Google Logging: ${GLOG_INCLUDE_DIRS}")
else (Glog_FOUND)
  message(FATAL_ERROR "Can't find Glog. Please set GLOG_INCLUDE_DIR & "
    "GLOG_LIBRARY")
endif (Glog_FOUND)

# Eigen
find_package(Eigen3 3 REQUIRED NO_MODULE)
if (EIGEN3_FOUND)
   message("-- Found Eigen3: ${EIGEN3_INCLUDE_DIRS}")
else (EIGEN3_FOUND)
   message(FATAL_ERROR "Cannot find Eigen3.")
endif (EIGEN3_FOUND)
include_directories("${CMAKE_CURRENT_SOURCE_DIR};${EIGEN3_INCLUDE_DIR}")

find_package(BLAS REQUIRED)

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
set(BOOST_ROOT "${MULTIVIEW_DEPS_DIR}")
set(Boost_DIR  "${MULTIVIEW_DEPS_DIR}")
set(BOOST_LIBRARYDIR "${MULTIVIEW_DEPS_DIR}/lib")
set(BOOST_INCLUDEDIR "${MULTIVIEW_DEPS_DIR}/include")
set(Boost_NO_SYSTEM_PATHS ON) # Do not search anywhere else
set(Boost_DEBUG OFF) # to help figure out issues
set(Boost_DETAILED_FAILURE_MSG OFF) # When on, helps with debugging
find_package(Boost REQUIRED COMPONENTS filesystem)
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

# Jasper, needed by OpenCV
find_package(Jasper REQUIRED)
if (JASPER_FOUND)
   message("-- Found Jasper: ${JASPER_INCLUDE_DIRS}")
else (JASPER_FOUND)
   message(FATAL_ERROR "Cannot find Jasper.")
endif (JASPER_FOUND)

# OpenCV
find_package(OpenCV 4.0.0 REQUIRED)
# Find OpenCV and fix a 3.3.1 bug
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

# Pcl
set(QHULL_ROOT "${MULTIVIEW_DEPS_DIR}") # so it does not pick system Qhull
find_package(PCL REQUIRED)
if (PCL_FOUND)
   message("-- Found PCL ${PCL_COMMON_INCLUDE_DIR}")
else (PCL_FOUND)
   message(FATAL_ERROR "Can't find PCL.")
endif (PCL_FOUND)
include_directories("${PCL_COMMON_INCLUDE_DIR}")

# CGAL
set(CGAL_TOOLS_INSTALL_DIR ${CMAKE_INSTALL_PREFIX})
add_subdirectory(cgal_tools)

# Build voxblox, which provides the voxblox_mesh tool for fusing point clouds into meshes
add_subdirectory(voxblox)

# Build mapmap
add_subdirectory(mapmap)

# Build mve
add_subdirectory(mve)

# Build rayint
add_subdirectory(rayint)

# Build texrecon (the repo name is mvs-texturing)
add_subdirectory(texrecon)

include_directories(${CMAKE_SOURCE_DIR}/rayint/libs
    ${CMAKE_SOURCE_DIR}/mve/libs
    ${CMAKE_SOURCE_DIR}/texrecon/libs
)

add_subdirectory(camera_model)
add_subdirectory(openMVG)
add_subdirectory(rig_calibrator)

set(BUILD_SHARED_LIBS ON CACHE INTERNAL "" FORCE)
set(USE_PYTHON OFF CACHE INTERNAL "" FORCE)
set(USE_OPENCV OFF CACHE INTERNAL "" FORCE)
set(USE_QT OFF CACHE INTERNAL "" FORCE)
set(USE_DICOM OFF CACHE INTERNAL "" FORCE)
set(USE_NUKE OFF CACHE INTERNAL "" FORCE)
set(USE_LIBRAW OFF CACHE INTERNAL "" FORCE)
set(OIIO_BUILD_TESTS OFF CACHE INTERNAL "" FORCE)
set(OIIO_BUILD_TOOLS OFF CACHE INTERNAL "" FORCE)
set(USE_OPENGL OFF CACHE INTERNAL "" FORCE)
set(BUILD_DOCS OFF CACHE INTERNAL "" FORCE)
set(INSTALL_DOCS OFF CACHE INTERNAL "" FORCE)
set(INSTALL_FONTS OFF CACHE INTERNAL "" FORCE)
set(OIIO_THREAD_ALLOW_DCLP OFF CACHE INTERNAL "" FORCE)
set(EMBEDPLUGINS OFF CACHE INTERNAL "" FORCE)
set(PROJECT_IS_TOP_LEVEL OFF CACHE INTERNAL "" FORCE)
set(USE_TBB OFF CACHE INTERNAL "" FORCE)
set(USE_FIELD3D OFF CACHE INTERNAL "" FORCE)
set(USE_OPENVDB OFF CACHE INTERNAL "" FORCE)
set(USE_QT OFF CACHE INTERNAL "" FORCE)
set(USE_OCIO OFF CACHE INTERNAL "" FORCE)
include_directories(${CMAKE_BINARY_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/OpenImageIO/src/include)
add_subdirectory(OpenImageIO)

set(BUILD_DOCUMENTATION OFF CACHE INTERNAL "" FORCE)
set(BUILD_TESTING OFF CACHE INTERNAL "" FORCE)
add_subdirectory(TheiaSfM)
