file(GLOB SOURCES "*.cc")
add_library(rig_calibrator_multiview SHARED ${SOURCES})

install(TARGETS rig_calibrator_multiview LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
