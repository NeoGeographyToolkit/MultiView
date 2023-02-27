# A few ultils from OpenMVG
file(GLOB SOURCES "*/*.c*")
add_library(rig_calibrator_openMVG SHARED ${SOURCES})

install(TARGETS rig_calibrator_openMVG LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
