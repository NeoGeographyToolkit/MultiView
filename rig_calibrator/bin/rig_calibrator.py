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
# TODO(oalexan1): Write here doc.
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
            raise Exception(
                "Could not make directory " + path + " as a file with this name exists."
            )

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
    print(" ".join(cmd) + "\n")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    process.wait()
    if process.returncode != 0:
        print("Failed execution of: " + " ".join(cmd))
        sys.exit(1)

def sanity_checks(undistort_image_path, import_map_path, build_map_path, args):

    # Check if the environment was set
    for var in [
        "ASTROBEE_RESOURCE_DIR",
        "ASTROBEE_CONFIG_DIR",
        "ASTROBEE_WORLD",
        "ASTROBEE_ROBOT",
    ]:
        if var not in os.environ:
            raise Exception("Must set " + var)

    if not os.path.exists(undistort_image_path):
        raise Exception("Executable does not exist: " + undistort_image_path)

    if not os.path.exists(import_map_path):
        raise Exception("Executable does not exist: " + import_map_path)

    if not os.path.exists(build_map_path):
        raise Exception("Executable does not exist: " + build_map_path)

    if args.image_list == "":
        raise Exception("The path to the output map was not specified.")

    if args.output_map == "":
        raise Exception("The path to the output map was not specified.")

    if args.work_dir == "":
        raise Exception("The path to the work directory was not specified.")

    if not os.path.exists(args.theia_flags):
        raise Exception("Cannot find the Theia flags file: " + args.theia_flags)

    if which("build_reconstruction") is None:
        raise Exception("Cannot find the 'build_reconstruction' program in PATH.")

    if which("export_to_nvm_file") is None:
        raise Exception("Cannot find the 'export_to_nvm_file' program in PATH.")

    if args.keep_undistorted_images and (not args.skip_rebuilding):
        raise Exception("Cannot rebuild the map if it has undistorted images.")


def process_args(args):
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

    parser.add_argument("--work_dir", dest="work_dir", default="",
                        help="A temporary work directory to be deleted by the user later.")
    
    args = parser.parse_args()

    # Print the help message if called with no arguments
    if num_input_args <= 1:
        parser.print_help()
        sys.exit(1)

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

# Theia likes its images undistorted and in one directory
def gen_undist_image_list(work_dir, dist_image_list):
    undist_dir = work_dir + "/undist"
    mkdir_p(undist_dir)

    # Wipe any preexisting images to not confuse theia later
    count = 0
    for image in glob.glob(undist_dir + "/*.jpg"):
        if count == 0:
            print("Wiping old images in: " + undist_dir)
        count += 1
        os.remove(image)

    undist_image_list = work_dir + "/undistorted_list.txt"
    print("Writing: " + undist_image_list)

    count = 10000  # to have the created images show up nicely
    undist_images = []
    with open(dist_image_list, "r") as dfh:
        dist_image_files = dfh.readlines()
        with open(undist_image_list, "w") as ufh:
            for image in dist_image_files:
                base_image = str(count) + ".jpg"
                undist_images.append(base_image)
                undist_image = undist_dir + "/" + base_image
                ufh.write(undist_image + "\n")
                count += 1

    return undist_image_list, undist_dir, undist_images


def genTheiaCalibFile(rig_config, args):

    # Parse the images for all cameras
    images = {}
    print("images are ", args.images)
    for image in args.images.split():
        print("--image is ", image)
        vals = image.split(':')
        cam_id = int(vals[0])
        pattern = vals[1]
        images[cam_id] = glob.glob(pattern)
        
        #print("---got ", cam_id, pattern)
        print("vals are ", cam_id, images[cam_id])

    print("Work dir ", args.work_dir)    
    mkdir_p(args.work_dir)

    calib_file = args.work_dir + "/" + "theia_calibration.json"

    print("--images ", images)
    print("Writing: " + calib_file)
    with open(calib_file, "w") as fh:
        fh.write("{\n")
        fh.write('"priors" : [\n')

        for cam_id in range(len(rig_config)):
            
            print("id is ", cam_id)
            print("---images ", images[cam_id])
            num_images = len(images[cam_id])
            
            for it in range(num_images):
                image = os.path.basename(images[cam_id][it])
                fh.write('{"CameraIntrinsicsPrior" : {\n')
                fh.write('"image_name" : "' + image + '",\n')
                fh.write('"width" : '  + rig_config[cam_id]['image_size'][0] + ",\n")
                fh.write('"height" : ' + rig_config[cam_id]['image_size'][1] + ",\n")
                fh.write('"camera_intrinsics_type" : "PINHOLE",\n')
                fh.write('"focal_length" : ' + rig_config[cam_id]["focal_length"] + ",\n")
                fh.write('"principal_point" : [' + \
                         rig_config[cam_id]["optical_center"][0] + ", " + \
                         rig_config[cam_id]["optical_center"][1] + "]\n")
                
                if it < num_images - 1 or cam_id < len(rig_config)  - 1:
                    fh.write("}},\n")
                else:
                    fh.write("}}\n")

        fh.write("]\n")
        fh.write("}\n")

    return calib_file

if __name__ == "__main__":

    args = process_args(sys.argv)

    rig_config = parseRigConfig(args.rig_config)

    calib_file = genTheiaCalibFile(rig_config, args)

    #print("rig config is ", rig_config)
    sys.exit(1)
    
    mkdir_p(args.work_dir)
    undist_image_list, undist_dir, undist_images = gen_undist_image_list(
        args.work_dir, args.image_list
    )

    # Undistort the images and crop to a central region
    undist_intrinsics_file = args.work_dir + "/undistorted_intrinsics.txt"
    cmd = [
        undistort_image_path,
        "-image_list",
        args.image_list,
        "--undistorted_crop_win",
        "1100 776",
        "-output_list",
        undist_image_list,
        "-undistorted_intrinsics",
        undist_intrinsics_file,
    ]
    run_cmd(cmd)

    (calib_file, intrinsics_str) = genTheiaCalibFile(args.work_dir, undist_images,
                                                     undist_intrinsics_file
    )
    recon_file = args.work_dir + "/run"
    matching_dir = args.work_dir + "/matches"

    # Wipe old data
    for old_recon in glob.glob(recon_file + "*"):
        print("Deleting old reconstruction: " + old_recon)
        os.remove(old_recon)

    count = 0
    for old_matches in glob.glob(matching_dir + "/*"):
        if count == 0:
            print("Wiping old matches in: " + matching_dir)
        count += 1
        os.remove(old_matches)

    cmd = [
        "build_reconstruction",
        "--flagfile",
        args.theia_flags,
        "--images",
        undist_dir + "/*jpg",
        "--calibration_file",
        calib_file,
        "--output_reconstruction",
        recon_file,
        "--matching_working_directory",
        matching_dir,
        "--intrinsics_to_optimize",
        "NONE",
    ]
    run_cmd(cmd)

    nvm_file = recon_file + ".nvm"
    cmd = [
        "export_to_nvm_file",
        "-input_reconstruction_file",
        recon_file + "-0",
        "-output_nvm_file",
        nvm_file,
    ]
    run_cmd(cmd)

    cmd = [
        import_map_path,
        "-input_map",
        nvm_file,
        "-output_map",
        args.output_map,
        "-undistorted_images_list",
        undist_image_list,
    ]

    if not args.keep_undistorted_images:
        cmd += [
            "-distorted_images_list",
            args.image_list,
        ]
    else:
        cmd += [
            "-undistorted_camera_params",
            intrinsics_str,
        ]

    run_cmd(cmd)

    if not args.skip_rebuilding:
        cmd = [
            build_map_path,
            "-output_map",
            args.output_map,
            "-rebuild",
            "-rebuild_refloat_cameras",
            "-rebuild_detector",
            "SURF",
            "--min_valid_angle",
            "1.0",  # to avoid features only seen in close-by images
        ]
    run_cmd(cmd)
