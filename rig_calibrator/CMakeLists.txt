# TODO(oalexan1): Must install the include files

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

# Build rig_calibrator lib
file(GLOB SOURCES "src/*.cc")
add_library(rig_calibrator SHARED ${SOURCES})
target_link_libraries(rig_calibrator 
    rig_calibrator_openMVG rig_calibrator_camera_model
    texture_reconstruction ${TEXRECON_LIBRARIES} ${OpenCV_LIBRARIES} ${CERES_LIBRARIES}
    ${JPEG_LIBRARIES} ${PNG_LIBRARIES} ${TIFF_LIBRARIES}
    TBB::tbb TBB::tbbmalloc mve mve_util
    gflags glog Boost::filesystem)

# Set RPATHS for the library
# TODO(oalexan1): Note sure about the robustness of the logic below but it seems to work
if(APPLE)
  set_target_properties(rig_calibrator PROPERTIES
    INSTALL_RPATH "@loader_path;${MULTIVIEW_DEPS_DIR}/lib")
elseif(UNIX) # Unix which is not Apple
  set_target_properties(rig_calibrator PROPERTIES
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib:${MULTIVIEW_DEPS_DIR}/lib")
endif()

# Build the rig_calibrator tool
add_executable(rig_calibrator_bin bin/rig_calibrator.cc)
set_target_properties(rig_calibrator_bin PROPERTIES OUTPUT_NAME "rig_calibrator")
target_link_libraries(rig_calibrator_bin 
    rig_calibrator)
set_target_properties(rig_calibrator_bin PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Set RPATHS for the tool. Setting it for the lib only as above is not enough.
# TODO(oalexan1): Note sure about the robustness of the logic below but it seems to work
if(APPLE)
  set_target_properties(rig_calibrator_bin PROPERTIES
    INSTALL_RPATH "@loader_path;${MULTIVIEW_DEPS_DIR}/lib")
elseif(UNIX) # Unix which is not Apple
  set_target_properties(rig_calibrator_bin PROPERTIES
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib:${MULTIVIEW_DEPS_DIR}/lib")
endif()

# Install the lib and the tool
install(TARGETS rig_calibrator LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS rig_calibrator_bin RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

