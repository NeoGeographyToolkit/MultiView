This software provides a collection of tools for joint calibration of
a set of rigs, each with one or more image and/or depth+image cameras,
fusion of point clouds into a mesh, and seamless texturing of meshes
with the input images and optimized cameras.

The software was originally developed as part of the [NASA
ISAAC](https://github.com/nasa/isaac#readme) project, which uses 
several robots with 3 front-facing cameras to scan and navigate the
International Space Station. The current version is self-contained
and does not depend on ROS or other parts of ISAAC.

The key dependencies are [Theia SfM](https://github.com/sweeneychris/TheiaSfM) 
for finding the initial camera poses, [Ceres Solver](http://ceres-solver.org/)
for solving the calibration problem, [VoxBlox](https://github.com/ethz-asl/voxblox)
for fusing point clouds into a mesh, and
[MVS Texturing](https://github.com/nmoehrle/mvs-texturing) for creation of
textured meshes. Most of the original work in this package went
towards the creation of the calibration logic and ensuring that all
components work together to create high-fidelity results.

These tools are shipped as part of the [NASA Ames Stereo
Pipeline](https://github.com/NeoGeographyToolkit/StereoPipeline/releases)
(for Linux and OSX).

# Documentation

 * [rig calibrator](https://stereopipeline.readthedocs.io/en/latest/tools/rig_calibrator.html)
 * [voxblox mesh creation](https://stereopipeline.readthedocs.io/en/latest/tools/voxblox_mesh.html)
 * [mesh texturing](https://stereopipeline.readthedocs.io/en/latest/tools/texrecon.html)
 * [stereo fusion](https://stereopipeline.readthedocs.io/en/latest/tools/multi_stereo.html)
# Fetching the code

# Build

It is suggested to fetch the binary versions of this package, per
above.

This package depends on other repositories, which are included as
submodules, and those may have their own dependencies. Hence, this
repo should be cloned recursively, as:

    git clone --recursive git@github.com:NeoGeographyToolkit/MultiView.git

Otherwise, after cloning it, run:

    git submodule update --init --recursive

Fetch the dependencies for this package with conda. For Linux, use:

    conda env create -f MultiView/conda/linux_deps_env_asp_3.2.0.yaml

while for OSX:

    conda env create -f MultiView/conda/osx_deps_env_asp_3.2.0.yaml

Then the software can be built as follows. Set the compilers. This is
different on Linux and OSX:

    isMac=$(uname -s | grep Darwin)
    if [ "$isMac" != "" ]; then
      cc_comp=clang
      cxx_comp=clang++
    else
      cc_comp=x86_64-conda_cos6-linux-gnu-gcc
      cxx_comp=x86_64-conda_cos6-linux-gnu-g++
    fi

Run ``cmake``:
    
    cd MultiView
    mkdir build
    cd build
    conda activate rig_calibrator
    toolsPath=$HOME/miniconda3/envs/rig_calibrator
    $toolsPath/bin/cmake ..                        \
      -DCMAKE_VERBOSE_MAKEFILE=TRUE                \
      -DMULTIVIEW_DEPS_DIR=$toolsPath              \
      -DCMAKE_INSTALL_PREFIX=$(pwd)/../install     \
      -DCMAKE_C_COMPILER=${toolsPath}/bin/$cc_comp \
      -DCMAKE_CXX_COMPILER=${toolsPath}/bin/$cxx_comp

Carefully check if all dependencies are found. If some are picked
not from the environment in $toolsPath, check your PATH and other
environmental variables, and remove from those the locations
which may tell ``cmake`` to look elsewhere. Then, run:

    make -j 20 && make install

The resulting tools will be installed in MultiView/install.

The Theia ``view_reconstruction`` tool can fail to build because of
its dependence on OpenGL, it can be excluded from building by editing
the appropriate CMakeLists.txt file.
