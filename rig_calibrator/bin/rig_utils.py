#!/usr/bin/python

import sys, os, re, subprocess

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
        
