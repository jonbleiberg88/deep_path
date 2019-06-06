#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 20:38:15 2019

Creates a qupath project with the annotations and images according to the 
naming conventions used in the project

@author: Jonathan Bleiberg
"""

import json
import os
import re
import shutil
import argparse
import time

ANNOTATION_DIR = "/Volumes/Elements/5:21:19/FL Annotations"
IMAGE_DIR = "/Volumes/Elements/5:21:19/FL Scans"
PROJECT_DIR = "/Volumes/Elements/5:21:19/Test_Proj"
PROJECT_FILE = "project.qpproj"



def create_qupath_proj(ANNOTATION_DIR, PROJECT_DIR, PROJECT_FILE, IMAGE_DIR):
    # extract file names of annotations
    annotation_files = [f for f in os.listdir(ANNOTATION_DIR) if f.endswith('.qpdata')]
    
    # create directories for project if not already exisiting
    if not os.path.isdir(os.path.join(PROJECT_DIR, "scripts")):
        os.mkdir(os.path.join(PROJECT_DIR, "scripts"))
    if not os.path.isdir(os.path.join(PROJECT_DIR, "thumbnails")):
        os.mkdir(os.path.join(PROJECT_DIR, "thumbnails"))
    if not os.path.isdir(os.path.join(PROJECT_DIR, "data")):
        os.mkdir(os.path.join(PROJECT_DIR, "data"))
    
    # copy annotation files to data folder in project directory
    for file in annotation_files:
            shutil.copy(os.path.join(ANNOTATION_DIR, file), os.path.join(PROJECT_DIR, "data",file))
    
    # if there already is a project, add annotation files to existing project
    if os.path.isfile(f"{PROJECT_DIR}/{PROJECT_FILE}"):
        with open(f"{PROJECT_DIR}/{PROJECT_FILE}") as f:
            json_dict = json.load(f)
            for file_name in annotation_files:
                json_dict['images'].append(generate_json_entry(file_name, IMAGE_DIR))
                
    # otherwise create a new project file with the annotations
    else:
        json_dict = {}
        now  = int(time.time() * 1000)
        json_dict["createTimestamp"] = now
        json_dict["modifyTimestamp"] = now
        json_dict["images"] = []
        for file_name in annotation_files:
                json_dict['images'].append(generate_json_entry(file_name, IMAGE_DIR))
        
        
    with open(f"{PROJECT_DIR}/{PROJECT_FILE}", 'w') as f:
        json.dump(json_dict, f, indent=4)


# create json entry for annotation according to naming pattern 
def generate_json_entry(file_name, IMAGE_DIR):
    name = file_name.replace(".qpdata", "")
    if "Scan" in name:
        path = f"{IMAGE_DIR}/" + "/".join(name.split("_")) + f"/{name}.qptiff"
    else:
        prefix = re.split("[0-9]+", name)[0]
        img_num = int(re.findall("[0-9]+", name)[0])

        path = f"{IMAGE_DIR}/{prefix}{img_num:02}/{name}.svs"
    entry = {'path': path,
             'name': name}
    return entry



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("project_dir", help="project root directory")
    parser.add_argument("annotation_dir", help="annotations root directory")
    parser.add_argument("image_dir", help="images root directory")
    parser.add_argument("-pf", "--project_file", help="optional project file name")
    args = parser.parse_args()
    
    if args.project_file:
        PROJECT_FILE = args.project_file
    
    create_qupath_proj(args.annotation_dir, args.project_dir, PROJECT_FILE, args.image_dir)
