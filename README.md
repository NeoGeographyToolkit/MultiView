# MultiView

This will be a collection of tools for calibration of a camera rig with image and depth+image sensors, creation of meshes from stereo and depth data, and seamless texturing.

# Fetching the code

This package depends on other repositories, which are included as
submodules, and those may have their own dependencies. Hence, this
repo should be cloned as::

    git clone --recursive git@github.com:oleg-alexandrov/MultiView.git

Otherwise, after cloning it, run::

    git submodule update --init --recursive

# Build

Change to that repository::

    cd MultiView

The dependencies for this package can be fetched with conda with the
command::

    conda env create -f conda/linux_deps_env.yaml

Then the software can be built (on Linux) as follows::

    mkdir build
    cd build
    conda activate rig_calibrator
    toolsPath=$HOME/miniconda3/envs/rig_calibrator
    $toolsPath/bin/cmake ..                                        \
      -DCMAKE_VERBOSE_MAKEFILE=TRUE                                \
      -DMULTIVIEW_DEPS_DIR=$toolsPath                              \
      -DCMAKE_INSTALL_PREFIX=$(pwd)/../install                     \
      -DCMAKE_C_COMPILER=$toolsPath/bin/x86_64-conda-linux-gnu-gcc \
      -DCMAKE_CXX_COMPILER=$toolsPath/bin/x86_64-conda-linux-gnu-c++
    make -j 20
    make install

The resulting tools will be installed in MultiView/install.

