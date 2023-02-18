#!/usr/bin/env python

"""
For a given list of topics and directory names, save the image and
point cloud messages for that topic to the specified directories. The
created file names will be in the form: <timestamp>.<extension>, with
the timestamp being the double-precision number of seconds since epoch
read from message header. Can specify a list of timestamps for which
to extract the data.
"""

import argparse, os, re, sys, rosbag, cv2, sensor_msgs, cv_bridge
from sensor_msgs import point_cloud2
import numpy as np

def read_timestamps(filename):
    """Return a map of timestamps."""

    timestamp_map = {}
    with open(filename, 'r') as f:
        lines = f.readlines()

    for line in lines:
        m = re.match("^.*?(\d+\.\d+).*?\n", line)
        if m:
            timestamp = float(m.group(1))
            timestamp_map[timestamp] = line
            
    return timestamp_map
  
parser = argparse.ArgumentParser(description = __doc__,
                                 formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--bag", dest = "bag", help = "Input bag."),
parser.add_argument("--topics", dest = "topics", default = "",  
                    help = "A list of topics, in quotes, having image, compressed image, and point cloud data.")
parser.add_argument("--dirs", dest = "dirs", default = "",
                    help = "A list of directories, in quotes, one for each topic, in which to save the data for that topic.")

parser.add_argument("--max", dest = "max", type = int, default = -1,
                    help = "The maximum number of messages to save. By default, save all of them.")

parser.add_argument("--timestamp_list", dest = "timestamp_list", default = "",  
                    help = "Extract data only for these timestamps. By default, extract all of them. Each line in this file must have a number of the form <digits>.<digits> or it will be ignored.")

args = parser.parse_args()

topic_list = args.topics.split()
dir_list = args.dirs.split()

# TODO(oalexan1): Make this a function.
# Sanity checks
if args.bag == "":
    print("Must specify the input bag.")
    sys.exit(1)
if len(topic_list) != len(dir_list):
    print("There must exists as many topics as output directories.")
    sys.exit(1)
if len(topic_list) == 0:
    print("Must specify at least one topic and output directory.")
    sys.exit(1)

timestamp_map = {}
if args.timestamp_list != "":
    timestamp_map = read_timestamps(args.timestamp_list)
    
# TODO(oalexan1): Make this a function.
# Map from topic name to dir name.
topic_to_dir = {}
for it in range(len(topic_list)):
    topic = topic_list[it]
    path  = dir_list[it]
    if topic in topic_to_dir:
        print("Duplicate topic: " + topic)
        sys.exit(1)
    topic_to_dir[topic] = path

# TODO(oalexan1): Make this a function.
# Create directories
for path in dir_list:
    try:
        os.makedirs(path)
    except OSError:
        if os.path.isdir(path):
            pass
        else:
            raise Exception("Could not make directory: " + path)

# TODO(oalexan1): Make this a function
# Read the bag
print("Reading: " + args.bag)
print("Writing the data to: " + args.dirs)

cv_bridge = cv_bridge.CvBridge()

# The timestamp format is consistent with what rig_calibrator uses
fmt_str = "{:10.7f}"

# Write the data
count = 0
with rosbag.Bag(args.bag, "r") as bag:

    info = bag.get_type_and_topic_info()

    # Check image message type
    msg_type = info.topics[topic].msg_type
    print("msgs type ", msg_type)
    
    for topic, msg, t in bag.read_messages(topic_list):

        # Read the header timestamp
        try:
            stamp = msg.header.stamp.to_sec()
            if len(timestamp_map) > 0 and (stamp not in timestamp_map):
                continue # if reading from a list
        except:
            continue

        if info.topics[topic].msg_type == "sensor_msgs/Image":
            # Write an image
            try:
                
                filename = topic_to_dir[topic] + "/" + fmt_str.format(stamp) + ".jpg"
                cv_image = cv_bridge.imgmsg_to_cv2(msg, desired_encoding = 'passthrough')
                if len(cv_image.shape) == 2:
                    # Try again, as mono. There should be a better way.
                    cv_image = cv_bridge.imgmsg_to_cv2(msg, desired_encoding = 'mono8')
                print("Writing: " + filename)
                cv2.imwrite(filename, cv_image)
                count += 1
            except:
                print("Failed to write: " + filename)

        if info.topics[topic].msg_type == "sensor_msgs/CompressedImage":
            # Write a compressed image. It is assumed to be in color.
            try:
                filename = topic_to_dir[topic] + "/" + fmt_str.format(stamp) + ".jpg"
                cv_image = cv_bridge.compressed_imgmsg_to_cv2(msg, desired_encoding = 'passthrough')
                print("Writing: " + filename)
                cv2.imwrite(filename, cv_image)
                count += 1
            except:
                print("Failed to write: " + filename)

        if info.topics[topic].msg_type == "sensor_msgs/PointCloud2":
            try:
                # Write a point cloud in the format at:
                # https://stereopipeline.readthedocs.io/en/latest/tools/rig_calibrator.html
                # (section on point cloud format). 
                # TODO(oalexan1): Test this!
                filename = topic_to_dir[topic] + "/" + fmt_str.format(stamp) + ".pc"
                print("Writing: " + filename)
                fh = open(filename, "wb")

                vals = np.array([msg.height, msg.width, 3], dtype=np.int32)
                fh.write(vals.tobytes())

                # TODO(oalexan1): This iteration may be slow.
                cloud_points = list(point_cloud2.read_points(msg,
                                                             skip_nans = False, # read all
                                                             field_names = ("x", "y", "z")))

                for point in cloud_points:
                    vals = np.array([point[0], point[1], point[2]], dtype=np.float32)
                    fh.write(vals.tobytes())

                count += 1
            except Exception as e:
                print("Failed to write: ", filename)
                print("Error is ", str(e))
        
        if args.max >= 0 and count >= args.max:
            print("Stopping at " + str(count) + " saved messages.")
            break
            
            
