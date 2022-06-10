# TODO(oalexan1): Reorganize the whole thing. In rig_calibrator must have src, bin, include

file(GLOB SOURCES "*.cc")
add_library(rig_calibrator_multiview SHARED ${SOURCES})
target_link_libraries(rig_calibrator_multiview 
    rig_calibrator_openMVG rig_calibrator_camera_model
    texture_reconstruction ${TEXRECON_LIBRARIES} ${OpenCV_LIBRARIES} ${CERES_LIBRARIES}
    ${JPEG_LIBRARIES} ${PNG_LIBRARIES} ${TIFF_LIBRARIES}
    TBB::tbb TBB::tbbmalloc mve mve_util
    gflags glog Boost::filesystem)

add_executable(rig_calibrator ${CMAKE_SOURCE_DIR}/bin/rig_calibrator.cc)
target_link_libraries(rig_calibrator 
    rig_calibrator_multiview rig_calibrator_openMVG rig_calibrator_camera_model
    texture_reconstruction ${TEXRECON_LIBRARIES} ${OpenCV_LIBRARIES} ${CERES_LIBRARIES}
    ${JPEG_LIBRARIES} ${PNG_LIBRARIES} ${TIFF_LIBRARIES}
    TBB::tbb TBB::tbbmalloc mve mve_util
    gflags glog Boost::filesystem)

# TODO(oalexan1): Note sure about the robustness of the logic below but it seems to work
if(APPLE)
  set_target_properties(rig_calibrator_multiview PROPERTIES
    INSTALL_RPATH "@loader_path;${MULTIVIEW_DEPS_DIR}/lib")
  set_target_properties(rig_calibrator PROPERTIES
    INSTALL_RPATH "@loader_path;${MULTIVIEW_DEPS_DIR}/lib")
elseif(UNIX) # Unix which is not Apple
  set_target_properties(rig_calibrator_multiview PROPERTIES
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib:${MULTIVIEW_DEPS_DIR}/lib")
  set_target_properties(rig_calibrator PROPERTIES
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib:${MULTIVIEW_DEPS_DIR}/lib")
endif()

# Create the executable in the bin bin directory
set_target_properties(rig_calibrator PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Install the lib and the tool
install(TARGETS rig_calibrator_multiview LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS rig_calibrator RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

