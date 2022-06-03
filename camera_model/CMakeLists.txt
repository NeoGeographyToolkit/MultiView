add_library(rig_calibrator_camera_model SHARED
   camera_params.cc camera_model.cc)

install(TARGETS rig_calibrator_camera_model LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

