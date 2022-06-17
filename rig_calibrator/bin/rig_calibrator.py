#!/usr/bin/env python
# Copyright (c) 2017, United States Government, as represented by the
# Administrator of the National Aeronautics and Space Administration.
#
# All rights reserved.
#
# The Astrobee platform is licensed under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with the
# License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

"""
# TODO(oalexan1): Write here documentation.
"""

import argparse, glob, os, re, shutil, subprocess, sys

def mkdir_p(path):
    if path == "":
        return  # this can happen when path is os.path.dirname("myfile.txt")
    try:
        os.makedirs(path)
    except OSError:
        if os.path.isdir(path):
            pass
        else:
            raise Exception("Could not make directory " +
                            path + " as a file with this name exists.")

def which(program):
    """Find if a program is in the PATH"""

    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None


def run_cmd(cmd):
    """
    Run a command.
    """
    # TODO(oalexan1): Should the "system" command be used instead?
    # Then it will print the output of each command.
    print(" ".join(cmd) + "\n")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    process.wait()
    if process.returncode != 0:
        print("Failed execution of: " + " ".join(cmd))
        sys.exit(1)

def sanity_checks(args):

    if args.rig_config == "":
        raise Exception("The path to the rig configuration file was not specified.")

    if args.images == "":
        raise Exception("The input images were not specified.")

    if args.theia_flags == "":
        raise Exception("The path to the Theia flags was not specified.")

    if not os.path.exists(args.theia_flags):
        raise Exception("Cannot find the Theia flags file: " + args.theia_flags)

    if args.work_dir == "":
        raise Exception("The path to the work directory was not specified.")

    if which("build_reconstruction") is None:
        raise Exception("Cannot find the 'build_reconstruction' program in PATH.")

    if which("export_to_nvm_file") is None:
        raise Exception("Cannot find the 'export_to_nvm_file' program in PATH.")

def processArgs(args):
    """
    Set up the parser and parse the args.
    """

    # Number of arguments before starting to parse them
    num_input_args = len(sys.argv)

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--rig_config",  dest="rig_config", default="",
                        help = "Rig configuration file.")

    parser.add_argument("--images",  dest="images", default="",
                        help = "Images (format tbd).")

    parser.add_argument("--theia_flags", dest="theia_flags",
                        default="", help="The flags to pass to Theia.")

    parser.add_argument("--work_dir", dest="work_dir", default="",
                        help="A temporary work directory to be deleted by the user later.")
    
    args = parser.parse_args()

    # Print the help message if called with no arguments
    if num_input_args <= 1:
        parser.print_help()
        sys.exit(1)

    sanity_checks(args)
    
    return args

def readConfigVals(handle, tag, num_vals):
    """
    Read a tag and vals. If num_vals > 0, expecting to read this many vals.
    """
    
    vals = []

    while True:

        line = handle.readline()
        if len(line) == 0:
            # Last line
            break

        if line[0] == '#':
            # Ignore comments
            continue

        line = line.rstrip() # wipe newlne
        if len(line) == 0:
            continue
        
        vals = line.split()
        if len(vals) == 0 or vals[0] != tag:
            raise Exception("Failed to read entry for: " + tag)

        vals = vals[1:]
        if num_vals > 0 and len(vals) != num_vals:
            raise Exception("Failed to read " + str(num_vals) + " values.")

        # Done
        break
    
    # Return the handle as well, as it changed
    return (vals, handle)

def parseRigConfig(rig_config_file):

    cameras = []
    
    with open(rig_config_file, "r") as handle:

        (vals, handle) = readConfigVals(handle, 'ref_sensor_id:', 1)
        ref_sensor_id = vals[0]

        while True:
            camera = {}
            
            (vals, handle) = readConfigVals(handle, 'sensor_id:', 1)
            if len(vals) == 0:
                break # end of file
            camera["sensor_id"] = vals[0]
            #print("sensor id ", camera["sensor_id"])

            (vals, handle) = readConfigVals(handle, "sensor_name:", 1)
            camera["sensor_name"] = vals[0]
            #print("sensor name ", camera["sensor_name"])

            (vals, handle) = readConfigVals(handle, "focal_length:", 1)
            camera["focal_length"] = vals[0]
            #print("focal_length ", camera["focal_length"])

            (vals, handle) = readConfigVals(handle, "optical_center:", 2)
            camera["optical_center"] = vals
            #print("optical_center ", camera["optical_center"])

            (vals, handle) = readConfigVals(handle, "distortion_coeffs:", -1) # var length
            camera["distortion_coeffs"] = vals
            #print("distortion_coeffs ", camera["distortion_coeffs"])
            if (len(vals) != 0 and len(vals) != 1 and len(vals) != 4 and len(vals) != 5):
                raise Exception("Expecting 0, 1, 4, or 5 distortion coefficients")

            (vals, handle) = readConfigVals(handle, "distortion_type:", 1)
            #print("vals is ", vals)
            if len(camera["distortion_coeffs"]) == 0 and vals[0] != "no_distortion":
                raise Exception("When there are no distortion coefficients, distortion type " + \
                                "must be: no_distortion")
            if len(camera["distortion_coeffs"]) == 1 and vals[0] != "fisheye":
                raise Exception("When there is 1 distortion coefficient, distortion type " + \
                                "must be: fisheye")
            if ((len(camera["distortion_coeffs"]) == 4 or len(camera["distortion_coeffs"]) == 5)
                and (vals[0] != "radtan")):
                raise Exception("When there are 4 or 5 distortion coefficients, distortion " + \
                                "type must be: radtan")
            camera["distortion_type"] = vals[0]

            (vals, handle) = readConfigVals(handle, "image_size:", 2)
            camera["image_size"] = vals
            #print("image_size ", camera["image_size"])
            
            (vals, handle) = readConfigVals(handle, "undistorted_image_size:", 2)
            camera["undistorted_image_size"] = vals
            #print("undistorted_image_size ", camera["undistorted_image_size"])

            (vals, handle) = readConfigVals(handle, "ref_to_sensor_transform:", -1)
            camera["ref_to_sensor_transform"] = vals
            #print("ref_to_sensor_transform ", camera["ref_to_sensor_transform"])
            
            (vals, handle) = readConfigVals(handle, "depth_to_image_transform:", -1)
            camera["depth_to_image_transform"] = vals
            #print("depth_to_image_transform ", camera["depth_to_image_transform"])

            (vals, handle) = readConfigVals(handle, "ref_to_sensor_timestamp_offset:", 1)
            camera["ref_to_sensor_timestamp_offset"] = vals[0]
            #print("ref_to_sensor_timestamp_offset ", camera["ref_to_sensor_timestamp_offset"])

            cameras.append(camera)
            
    return cameras

def imageExtension(images):
    '''
    Get the image extensions for all images. Check that there is only one.
    '''
    
    extensions = set()
    for image in images:
        path, ext = os.path.splitext(image)
        print("--add ", ext)
        extensions.add(ext)
    if len(extensions) > 1:
        raise Exception("Input images have a mix of filename extensions. Use just one.")
    if len(extensions) == 0:
        raise Exception("The input image set is invalid.")
    print("--will get ", list(extensions)[0])
    return list(extensions)[0]
        
def genTheiaCalibFile(rig_config, args):

    # Parse the images for all cameras
    images = {}
    extensions = set()
    print("images are ", args.images)
    for image in args.images.split():
        vals = image.split(':')
        cam_id = int(vals[0])
        pattern = vals[1]
        images[cam_id] = glob.glob(pattern)
        ext = imageExtension(images[cam_id])
        extensions.add(ext)
        
    if len(extensions) > 1:
        raise Exception("Input images have a mix of filename extensions. Use just one.")
    if len(extensions) == 0:
        raise Exception("The input image set is invalid.")
    extension = list(extensions)[0]

    print("Work directory: " + args.work_dir)    
    mkdir_p(args.work_dir)

    # Remove old images in sym_image_dir
    sym_image_dir = args.work_dir + "/images"
    old_images = glob.glob(sym_image_dir + "/*")
    if len(old_images) > 0:
        print("Removing old images from " + sym_image_dir)
        for image in old_images:
            os.remove(image)

    # Theia likes all images in the same dir, so do it with sym links
    print("Creating sym links to the input images in: " + sym_image_dir)
    mkdir_p(sym_image_dir)
    sym_images = {}
    for cam_id in range(len(rig_config)):
        sym_images[cam_id] = []
        num_images = len(images[cam_id])
        for it in range(num_images):
            image = images[cam_id][it]
            src_file = os.path.relpath(image, sym_image_dir)
            cam_type = os.path.basename(os.path.dirname(image))
            dst_file = sym_image_dir + "/" + cam_type + "_" + os.path.basename(image)
            
            sym_images[cam_id].append(dst_file)
            os.symlink(src_file, dst_file)
    
    calib_file = args.work_dir + "/" + "theia_calibration.json"
    print("Writing Theia calibration file: " + calib_file)
    with open(calib_file, "w") as fh:
        fh.write("{\n")
        fh.write('"priors" : [\n')

        for cam_id in range(len(rig_config)):
            print("rig is ", rig_config[cam_id])
            
            num_images = len(sym_images[cam_id])
            for it in range(num_images):
                image = os.path.basename(sym_images[cam_id][it])
                fh.write('{"CameraIntrinsicsPrior" : {\n')
                fh.write('"image_name" : "' + image + '",\n')
                fh.write('"width" : '  + rig_config[cam_id]['image_size'][0] + ",\n")
                fh.write('"height" : ' + rig_config[cam_id]['image_size'][1] + ",\n")
                fh.write('"focal_length" : ' + rig_config[cam_id]["focal_length"] + ",\n")
                fh.write('"principal_point" : [' + \
                         rig_config[cam_id]["optical_center"][0] + ", " + \
                         rig_config[cam_id]["optical_center"][1] + "],\n")

                if rig_config[cam_id]['distortion_type'] == 'no_distortion':
                    fh.write('"camera_intrinsics_type" : "PINHOLE"\n')
                elif rig_config[cam_id]['distortion_type'] == 'fisheye':
                    fh.write('"radial_distortion_1" : ' + \
                             rig_config[cam_id]["distortion_coeffs"][0] + ",\n")
                    fh.write('"camera_intrinsics_type" : "FOV"\n')
                elif rig_config[cam_id]['distortion_type'] == 'radtan':
                    
                    # Distortion coeffs convention copied from
                    # camera_params.cc. JSON format from
                    # calibration_test.json in TheiaSFM.
                    k1 = rig_config[cam_id]["distortion_coeffs"][0]
                    k2 = rig_config[cam_id]["distortion_coeffs"][1]
                    p1 = rig_config[cam_id]["distortion_coeffs"][2]
                    p2 = rig_config[cam_id]["distortion_coeffs"][3]
                    k3 = '0'
                    if len(rig_config[cam_id]["distortion_coeffs"]) == 5:
                        k3 = rig_config[cam_id]["distortion_coeffs"][4]
                    fh.write('"radial_distortion_coeffs" : [' + \
                             k1 + ", " + k2 + ", " + k3 + "],\n")
                    fh.write('"tangential_distortion_coeffs" : [' + \
                             p1 + ", " + p2 + "],\n")
                    fh.write('"camera_intrinsics_type" : "PINHOLE_RADIAL_TANGENTIAL"\n')
                else:
                    raise Exception("Unknown distortion type: " + \
                                    rig_config[cam_id]['distortion_type'])

                if it < num_images - 1 or cam_id < len(rig_config)  - 1:
                    fh.write("}},\n")
                else:
                    fh.write("}}\n")

        fh.write("]\n")
        fh.write("}\n")

    return (calib_file, sym_image_dir, images, sym_images, extension)

def put_orig_images_in_nvm(nvm_file, orig_nvm_file, images, sym_images):
    """
    Theia saves images without full path. Go back to original image names.
    """
    # Make a dict for quick lookup
    image_dict = {}
    for key in images:
        for i in range(len(images[key])):
            image_dict[os.path.basename(sym_images[key][i])] = images[key][i]
        
    lines = []
    with open(nvm_file, 'r') as fh:
        lines = fh.readlines()

    for it in range(len(lines)):
        vals = lines[it].split()
        if len(vals) > 0 and vals[0] in image_dict:
            vals[0] = image_dict[vals[0]]
            lines[it] = " ".join(vals) + "\n"

    print("Writing nvm file with original image names: " + orig_nvm_file)
    with open(orig_nvm_file, 'w') as fh:
        fh.writelines(lines)
    
if __name__ == "__main__":

    args = processArgs(sys.argv)

    rig_config = parseRigConfig(args.rig_config)

    (calib_file, sym_image_dir, images, sym_images, image_extension) \
                 = genTheiaCalibFile(rig_config, args)

    print("Must set: export OIIO_LIBRARY_PATH=$HOME/projects/MultiView/install/lib")
    print("Must remove this dependency,")
    
    reconstruction_file = args.work_dir + "/reconstruction"
    matching_dir = args.work_dir + "/matches"

    # Wipe old data
    for old_reconstruction in glob.glob(reconstruction_file + "*"):
        print("Deleting old reconstruction: " + old_reconstruction)
        os.remove(old_reconstruction)

    count = 0
    for old_matches in glob.glob(matching_dir + "/*"):
        if count == 0:
            print("Wiping old matches in: " + matching_dir)
        count += 1
        os.remove(old_matches)

    cmd = ["build_reconstruction", "--flagfile", args.theia_flags, "--images",
           sym_image_dir + "/*" + image_extension, "--calibration_file", calib_file,
           "--output_reconstruction", reconstruction_file, "--matching_working_directory",
           matching_dir, "--intrinsics_to_optimize", "NONE"]
    run_cmd(cmd)
    
    nvm_file = reconstruction_file + ".nvm"
    cmd = ["export_to_nvm_file", "-input_reconstruction_file",  reconstruction_file + "-0",
           "-output_nvm_file", nvm_file]
    run_cmd(cmd)

    orig_nvm_file = reconstruction_file + "_orig.nvm"
    put_orig_images_in_nvm(nvm_file, orig_nvm_file, images, sym_images)
    
