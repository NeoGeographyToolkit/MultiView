# TODO(oalexan1): Must install the include files

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

# Build rig_calibrator lib
file(GLOB SOURCES "src/*.cc")
add_library(rig_calibrator_lib SHARED ${SOURCES})
target_link_libraries(rig_calibrator_lib 
    rig_calibrator_openMVG rig_calibrator_camera_model
    texture_reconstruction ${TEXRECON_LIBRARIES} ${OpenCV_LIBRARIES} ${CERES_LIBRARIES}
    ${JPEG_LIBRARIES} ${PNG_LIBRARIES} ${TIFF_LIBRARIES}
    TBB::tbb TBB::tbbmalloc mve mve_util
    gflags glog Boost::filesystem pcl_common pcl_io_ply pcl_io)

# Build the rig_calibrator tool
add_executable(rig_calibrator_bin bin/rig_calibrator.cc)
set_target_properties(rig_calibrator_bin PROPERTIES OUTPUT_NAME "rig_calibrator")
target_link_libraries(rig_calibrator_bin 
    rig_calibrator_lib)
set_target_properties(rig_calibrator_bin PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Build the undistort_image_texrecon auxiliary tool
add_executable(undistort_image_texrecon bin/undistort_image_texrecon.cc)
target_link_libraries(undistort_image_texrecon 
    rig_calibrator_lib)
set_target_properties(undistort_image_texrecon PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Build the fit_rpc auxiliary tool
add_executable(fit_rpc bin/fit_rpc.cc)
target_link_libraries(fit_rpc 
    rig_calibrator_lib)
set_target_properties(fit_rpc PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Build the sfm_submap tool
add_executable(sfm_submap bin/sfm_submap.cc)
target_link_libraries(sfm_submap 
    rig_calibrator_lib)
set_target_properties(sfm_submap PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Build the sfm_merge tool
add_executable(sfm_merge bin/sfm_merge.cc)
target_link_libraries(sfm_merge 
    rig_calibrator_lib)
set_target_properties(sfm_merge PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Set RPATHS. 
# TODO(oalexan1): Use a loop here, as there are too many lines
if(APPLE)
  set_target_properties(rig_calibrator_lib PROPERTIES
    INSTALL_RPATH "@loader_path;@loader_path/../lib;${MULTIVIEW_DEPS_DIR}/lib")
  set_target_properties(rig_calibrator_bin PROPERTIES
    INSTALL_RPATH "@loader_path;@loader_path/../lib;${MULTIVIEW_DEPS_DIR}/lib")
  set_target_properties(undistort_image_texrecon PROPERTIES
    INSTALL_RPATH "@loader_path;@loader_path/../lib;${MULTIVIEW_DEPS_DIR}/lib")
  set_target_properties(fit_rpc PROPERTIES
    INSTALL_RPATH "@loader_path;@loader_path/../lib;${MULTIVIEW_DEPS_DIR}/lib")
  set_target_properties(sfm_submap PROPERTIES
    INSTALL_RPATH "@loader_path;@loader_path/../lib;${MULTIVIEW_DEPS_DIR}/lib")
  set_target_properties(sfm_merge PROPERTIES
    INSTALL_RPATH "@loader_path;@loader_path/../lib;${MULTIVIEW_DEPS_DIR}/lib")
elseif(UNIX) # Unix which is not Apple
  set_target_properties(rig_calibrator_lib PROPERTIES
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib:${MULTIVIEW_DEPS_DIR}/lib")
  set_target_properties(rig_calibrator_bin PROPERTIES
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib:${MULTIVIEW_DEPS_DIR}/lib")
  set_target_properties(undistort_image_texrecon PROPERTIES
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib:${MULTIVIEW_DEPS_DIR}/lib")
  set_target_properties(fit_rpc PROPERTIES
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib:${MULTIVIEW_DEPS_DIR}/lib")
  set_target_properties(sfm_submap PROPERTIES
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib:${MULTIVIEW_DEPS_DIR}/lib")
  set_target_properties(sfm_merge PROPERTIES
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib:${MULTIVIEW_DEPS_DIR}/lib")
endif()

# Install the lib and the tools
install(TARGETS rig_calibrator_lib LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS rig_calibrator_bin RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(TARGETS undistort_image_texrecon RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(TARGETS fit_rpc RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(TARGETS sfm_submap RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(TARGETS sfm_merge RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# Install the python tools.
# TODO(oalexan1): Make a loop here.
INSTALL(FILES bin/theia_sfm
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_EXECUTE GROUP_READ WORLD_EXECUTE
    DESTINATION bin)
INSTALL(FILES bin/texrecon
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_EXECUTE GROUP_READ WORLD_EXECUTE
    DESTINATION bin)
INSTALL(FILES bin/multi_stereo
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_EXECUTE GROUP_READ WORLD_EXECUTE
    DESTINATION bin)

# Install the python tool utils
INSTALL(FILES bin/rig_utils.py
    PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ
    DESTINATION libexec)

# Install the Theia flags
INSTALL(FILES theia_flags.txt
    PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ
    DESTINATION share)

