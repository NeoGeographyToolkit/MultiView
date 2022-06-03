# A few ultils from OpenMVG
add_library(rig_calibrator_openMVG SHARED
    multiview/projection.cc multiview/triangulation_nview.cc numeric/nullspace.cc)

install(TARGETS rig_calibrator_openMVG LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
