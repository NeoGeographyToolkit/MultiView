#!/usr/bin/python

import sys, os, re, subprocess, shutil, subprocess, glob
import numpy as np

if sys.version_info < (3, 0, 0):
    print('\nERROR: Must use Python 3.0 or greater.')
    sys.exit(1)

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

def add_missing_quotes(vals):
    """
    Given a list of strings, protect each one having spaces with quotes, if not present.
    """

    out_vals = []
    for val in vals:
        if (' ' in val or '\t' in val) and (val[0] != '\'' and val[0] != '"'):
            val = '\'' + val + '\''
        out_vals.append(val)
    return out_vals
        
def run_cmd(cmd, quit_on_failure = True):
    """
    Run a command. Print the output in real time.
    """

    # This is a bugfix for not printing quotes around strings with spaces
    fmt_cmd = add_missing_quotes(cmd)
    print(" ".join(fmt_cmd))

    # Note how we redirect stderr to stdout
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         universal_newlines=True)

    # Print the standard output in real time
    status = 0
    while True:
        out = p.stdout.readline()
        if not out:
            break # finished the run

        # Print the line but wipe the extra whitespace
        print(out.rstrip())

    status = p.poll()

    # Sometimes status is None which appears to be a bug in Python.
    # Let that pass. Only if the status is a valid integer and non-zero,
    # then quit on failure if requested to do so.
    if status is not None and status != 0 and quit_on_failure:
        print("Failed execution of: " + " ".join(cmd) + " with status " + str(status))
        sys.exit(1)

    return status

def readConfigVals(handle, tag, num_vals):
    """
    Read a tag and vals. If num_vals > 0, expecting to read this many vals.
    """
    
    vals = []

    while True:

        line = handle.readline()
        
        # Wipe everything after pound but keep the newline,
        # as otherwise this will be treated as the last line
        match = re.match("^(.*?)\#.*?\n", line)
        if match:
            line = match.group(1) + "\n"

        if len(line) == 0:
            # Last line, lacks a newline
            break

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

        (vals, handle) = readConfigVals(handle, 'ref_sensor_name:', 1)
        ref_sensor_name = vals[0]

        while True:
            camera = {}
            
            (vals, handle) = readConfigVals(handle, 'sensor_name:', 1)
            if len(vals) == 0:
                break # end of file
            camera["sensor_name"] = vals[0]

            (vals, handle) = readConfigVals(handle, "focal_length:", 1)
            camera["focal_length"] = vals[0]

            (vals, handle) = readConfigVals(handle, "optical_center:", 2)
            camera["optical_center"] = vals

            (vals, handle) = readConfigVals(handle, "distortion_coeffs:", -1) # var length
            camera["distortion_coeffs"] = vals
            if (len(vals) != 0 and len(vals) != 1 and len(vals) != 4 and len(vals) != 5):
                raise Exception("Expecting 0, 1, 4, or 5 distortion coefficients")

            (vals, handle) = readConfigVals(handle, "distortion_type:", 1)
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
            
            (vals, handle) = readConfigVals(handle, "distorted_crop_size:", 2)
            camera["distorted_crop_size"] = vals

            (vals, handle) = readConfigVals(handle, "undistorted_image_size:", 2)
            camera["undistorted_image_size"] = vals

            (vals, handle) = readConfigVals(handle, "ref_to_sensor_transform:", -1)
            camera["ref_to_sensor_transform"] = vals
            
            (vals, handle) = readConfigVals(handle, "depth_to_image_transform:", -1)
            camera["depth_to_image_transform"] = vals

            (vals, handle) = readConfigVals(handle, "ref_to_sensor_timestamp_offset:", 1)
            camera["ref_to_sensor_timestamp_offset"] = vals[0]

            cameras.append(camera)

    return cameras

def imageExtension(images):
    '''
    Get the image extensions for all images. Check that there is only one.
    '''
    
    extensions = set()
    for image in images:
        path, ext = os.path.splitext(image)
        extensions.add(ext)
    if len(extensions) > 1:
        raise Exception("Input images have a mix of filename extensions. Use just one.")
    if len(extensions) == 0:
        raise Exception("The input image set is invalid.")
    return list(extensions)[0]
        
def parse_images_and_camera_poses(image_list, rig_sensor,
                                  # These indices will start from 1, if specified
                                  first_image_index = None, last_image_index = None):

    with open(image_list, 'r') as f:
        lines = f.readlines()

    images = []
    world_to_cam = []
    image_count = 0 # Below, the count of the first image will be 1
    for line in lines:
        m = re.match("^(.*?)\#", line)
        if m:
            # Wipe comments
            line = m.group(1)
        line = line.rstrip()
        if len(line) == 0:
            continue
        
        vals = line.split()
        if len(vals) < 13:
            raise Exception("Could not parse: " + image_list)

        image = vals[0]
        vals = vals[1:13]
        
        curr_sensor = os.path.basename(os.path.dirname(image))
        if curr_sensor != rig_sensor:
            continue

        image_count += 1

        # If to use only a range
        if first_image_index is not None and last_image_index is not None:
            if image_count < first_image_index or image_count > last_image_index:
                continue
        
        # Put the values in a matrix
        M = np.identity(4)
        val_count = 0
        # Read rotation
        for row in range(3):
            for col in range(3):
                M[row][col] = float(vals[val_count])
                val_count = val_count + 1
        # Read translation
        for row in range(3):
            M[row][3] = float(vals[val_count])
            val_count = val_count + 1

        images.append(image)
        world_to_cam.append(M)
        
    return (images, world_to_cam)

def undistort_images(args, images, tools_base_dir, extension, extra_opts, suff):

    # Form the list of distorted images
    dist_image_list = args.out_dir + "/" + args.rig_sensor + "/distorted_index.txt"
    mkdir_p(os.path.dirname(dist_image_list))
    print("Writing: " + dist_image_list)
    dist_images = []
    with open(dist_image_list, 'w') as f:
        for image in images:
            dist_images.append(image)
            f.write(image + "\n")

    # Form the list of unundistorted images
    undist_dir = args.out_dir + "/" + args.rig_sensor + "/undistorted" + suff

    if os.path.isdir(undist_dir):
        # Wipe the existing directory, as it may have stray files
        print("Removing recursively old directory: " + undist_dir)
        shutil.rmtree(undist_dir)
    
    undist_image_list = undist_dir + "/index.txt"
    mkdir_p(undist_dir)
    print("Writing: " + undist_image_list)
    undistorted_images = []
    with open(undist_image_list, 'w') as f:
        for image in dist_images:
            image = undist_dir + "/" + os.path.basename(image)
            # Use desired extension. For example, texrecon seems to want .jpg. In stereo
            # one prefers .tif, as that one is lossless.
            path, ext = os.path.splitext(image)
            image = path + extension
            undistorted_images.append(image)
            f.write(image + "\n")

    undist_intrinsics = undist_dir + "/undistorted_intrinsics.txt"
    cmd = [tools_base_dir + "/bin/undistort_image_texrecon",
           "--image_list", dist_image_list,
           "--output_list", undist_image_list,
           "--rig_config", args.rig_config,
           "--rig_sensor", args.rig_sensor,
           "--undistorted_crop_win", args.undistorted_crop_win,
           "--undistorted_intrinsics", undist_intrinsics] + \
           extra_opts
    
    print("Undistorting " + args.rig_sensor + " images.")
    run_cmd(cmd)

    return (undist_intrinsics, undistorted_images, undist_dir)

def read_intrinsics(intrinsics_file):
    
    if not os.path.exists(intrinsics_file):
        raise Exception("Missing file: " + intrinsics_file)

    with open(intrinsics_file, "r") as f:
        for line in f:
            if re.match("^\s*\#", line):
                continue  # ignore the comments
            vals = line.split()
            if len(vals) < 5:
                raise Exception("Expecting 5 parameters in " + intrinsics_file)
            widx = float(vals[0])
            widy = float(vals[1])
            f = float(vals[2])
            cx = float(vals[3])
            cy = float(vals[4])

            return(widx, widy, f, cx, cy)

    # If no luck
    raise Exception("Could not read intrinsics from: " + intrinsics_file)

def write_tsai_camera_file(tsai_file, f, cx, cy, cam_to_world):
    """
    Write a tsai camera file understandable by ASP. Assume that the
    intrinsics f, cx, cy are in units of pixels, and the pitch is 1,
    and that there is no distortion.
    """

    print("Writing: " + tsai_file)
    M = cam_to_world # to save on typing
    with open(tsai_file, "w") as g:
        g.write("VERSION_3\n")
        g.write("fu = %0.17g\n" % f)
        g.write("fv = %0.17g\n" % f)
        g.write("cu = %0.17g\n" % cx)
        g.write("cv = %0.17g\n" % cy)
        g.write("u_direction = 1 0 0\n")
        g.write("v_direction = 0 1 0\n")
        g.write("w_direction = 0 0 1\n")
        g.write("C = %0.17g %0.17g %0.17g\n" % (M[0][3], M[1][3], M[2][3]))
        g.write("R = %0.17g %0.17g %0.17g %0.17g %0.17g %0.17g %0.17g %0.17g %0.17g\n" %
                (M[0][0], M[0][1], M[0][2],
                 M[1][0], M[1][1], M[1][2],
                 M[2][0], M[2][1], M[2][2]))
        g.write("pitch = 1\n")
        g.write("NULL\n")

def write_cam_to_world_matrix(cam_to_world_file, cam_to_world):

    print("Writing: " + cam_to_world_file)
    M = cam_to_world # to save on typing
    with open(cam_to_world_file, "w") as g:
        g.write(("%0.17g %0.17g %0.17g %0.17g\n" + \
                 "%0.17g %0.17g %0.17g %0.17g\n" + \
                 "%0.17g %0.17g %0.17g %0.17g\n" + \
                 "%0.17g %0.17g %0.17g %0.17g\n") % 
                (M[0][0], M[0][1], M[0][2], M[0][3],
                 M[1][0], M[1][1], M[1][2], M[1][3],
                 M[2][0], M[2][1], M[2][2], M[2][3],
                 M[3][0], M[3][1], M[3][2], M[3][3]))
