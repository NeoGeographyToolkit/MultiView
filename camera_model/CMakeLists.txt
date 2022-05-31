message(STATUS "Now in camera model4")

add_library(rig_camera_model SHARED
    camera_params.cc camera_model.cc)
