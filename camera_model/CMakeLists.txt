file(GLOB SOURCES "*.cc")

add_library(rig_calibrator_camera_model SHARED ${SOURCES})
target_link_libraries(rig_calibrator_camera_model 
    ${CERES_LIBRARIES} glog)

install(TARGETS rig_calibrator_camera_model LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

