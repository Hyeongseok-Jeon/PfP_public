#!/usr/bin/env python

"""A simple python script to generate sequence videos."""

import argparse
import os
import re
import shutil
import sys
from collections import defaultdict
from typing import List
import random
import matplotlib.animation as anim
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.visualization.visualize_sequences import viz_sequence

def viz_bbox(agent, cur_x, cur_y, cur_heading):
    length = 4
    width = 1.8

    if agent == 'AV':
        color = "#007672"
    elif agent == 'AGENT':
        color = "#d33e4c"
    elif agent == 'OTHERS':
        color = "#d3e8ef"
        
    px = []
    py = []
    
    px.append(cur_x + 0.5*length*np.cos(cur_heading) + 0.5 * width * np.sin(cur_heading))
    px.append(cur_x + 0.5*length*np.cos(cur_heading) - 0.5 * width * np.sin(cur_heading))
    px.append(cur_x - 0.5*length*np.cos(cur_heading) - 0.5 * width * np.sin(cur_heading))
    px.append(cur_x - 0.5*length*np.cos(cur_heading) + 0.5 * width * np.sin(cur_heading))
    px.append(cur_x + 0.5*length*np.cos(cur_heading) + 0.5 * width * np.sin(cur_heading))

    py.append(cur_y + 0.5*length*np.sin(cur_heading) - 0.5 * width * np.cos(cur_heading))
    py.append(cur_y + 0.5*length*np.sin(cur_heading) + 0.5 * width * np.cos(cur_heading))
    py.append(cur_y - 0.5*length*np.sin(cur_heading) + 0.5 * width * np.cos(cur_heading))
    py.append(cur_y - 0.5*length*np.sin(cur_heading) - 0.5 * width * np.cos(cur_heading))
    py.append(cur_y + 0.5*length*np.sin(cur_heading) - 0.5 * width * np.cos(cur_heading))
    
    return px, py, color
    
output_dir = '/home/jhs/Desktop/research2_fulldata/argoverse/visualization'
root = '/home/jhs/Desktop/research2_fulldata/argoverse/forecasting_train_v1.1/train/data'
sequence_list = os.listdir(root)


for seq_name in range(15):
    seq_idx = random.randint(0, len(sequence_list))
    seq_name = sequence_list[seq_idx]
    seq_path = os.path.join(root, seq_name)
    df = pd.read_csv(seq_path)
    count = 0
    time_list = np.sort(np.unique(df["TIMESTAMP"].values))

    # Get API for Argo Dataset map
    avm = ArgoverseMap()
    city_name = df["CITY_NAME"].values[0]
    seq_lane_bbox = avm.city_halluc_bbox_table[city_name]
    seq_lane_props = avm.city_lane_centerlines_dict[city_name]
    

    df_av = df[df["OBJECT_TYPE"] == 'AV']
    x_min_av = np.min(df_av["X"])
    x_max_av = np.max(df_av["X"])
    y_min_av = np.min(df_av["Y"])
    y_max_av = np.max(df_av["Y"])
    df_agent = df[df["OBJECT_TYPE"] == 'AGENT']
    x_min_agent = np.min(df_agent["X"])
    x_max_agent = np.max(df_agent["X"])
    y_min_agent = np.min(df_agent["Y"])
    y_max_agent = np.max(df_agent["Y"])
    
    x_min = np.min((x_min_av, x_min_agent)) - 25
    x_max = np.max((x_min_av, x_min_agent)) + 25
    y_min = np.min((y_min_av, y_min_agent)) - 25
    y_max = np.max((y_max_av, y_max_agent)) + 25
    
    x_min_fov = min(df["X"])
    x_max_fov = max(df["X"])
    y_min_fov = min(df["Y"])
    y_max_fov = max(df["Y"])
    lane_centerlines = []
    # Get lane centerlines which lie within the range of trajectories
    for lane_id, lane_props in seq_lane_props.items():

        lane_cl = lane_props.centerline

        if (
            np.min(lane_cl[:, 0]) < x_max_fov
            and np.min(lane_cl[:, 1]) < y_max_fov
            and np.max(lane_cl[:, 0]) > x_min_fov
            and np.max(lane_cl[:, 1]) > y_min_fov
        ):
            lane_centerlines.append(lane_cl)

    seq_out_dir = os.path.join(output_dir, seq_name.split(".")[0])
    

    
    for stamp in range(40):
        cur_time = time_list[stamp]
        end_time = time_list[stamp+10]

        df_cur = df.loc[df["TIMESTAMP"] >= cur_time]
        df_cur = df_cur[df_cur["TIMESTAMP"] <= end_time]
        viz_sequence(
            df_cur,
            lane_centerlines=lane_centerlines,
            show=False,
            smoothen=False,
        )
        
        
        df_cur_head = df_cur[df_cur["TIMESTAMP"] == cur_time]
        df_cur_head = df_cur_head[df_cur_head["OBJECT_TYPE"] == "AV"]
        head_vector = avm.get_lane_direction(np.asarray([float(df_cur_head['X']), float(df_cur_head['Y'])]), city_name)[0]
        x,y,c = viz_bbox("AV", float(df_cur_head["X"]), float(df_cur_head["Y"]), np.arctan2(head_vector[1], head_vector[0]))
        plt.plot(x,y,color=c)
        
        df_cur_head = df_cur[df_cur["TIMESTAMP"] == cur_time]
        df_cur_head = df_cur_head[df_cur_head["OBJECT_TYPE"] == "AGENT"]
        head_vector = avm.get_lane_direction(np.asarray([float(df_cur_head['X']), float(df_cur_head['Y'])]), city_name)[0]
        x,y,c = viz_bbox("AGENT", float(df_cur_head["X"]), float(df_cur_head["Y"]), np.arctan2(head_vector[1], head_vector[0]))
        plt.plot(x,y,color=c)
        plt.axis('equal')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        
        df_cur_head = df_cur[df_cur["TIMESTAMP"] == cur_time]
        df_cur_head = df_cur_head[df_cur_head["OBJECT_TYPE"] == "OTHERS"]
        for other_veh in range(len(df_cur_head)):
             head_vector = avm.get_lane_direction(np.asarray([float(df_cur_head[other_veh:other_veh+1]['X']), float(df_cur_head[other_veh:other_veh+1]['Y'])]), city_name)[0]
             x,y,c = viz_bbox("OTHERS", float(df_cur_head[other_veh:other_veh+1]["X"]), float(df_cur_head[other_veh:other_veh+1]["Y"]), np.arctan2(head_vector[1], head_vector[0]))
             plt.plot(x,y,color=c)
        
        os.makedirs(seq_out_dir, exist_ok=True)

        plt.savefig(
            os.path.join(seq_out_dir, f"{count}.png"),
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()
        count += 1

    from moviepy.editor import ImageSequenceClip

    img_idx = sorted([int(x.split(".")[0]) for x in os.listdir(seq_out_dir)])
    list_video = [f"{seq_out_dir}/{x}.png" for x in img_idx]
    clip = ImageSequenceClip(list_video, fps=10)
    video_path = os.path.join(output_dir, f"{seq_name.split('.')[0]}.mp4")
    clip.write_videofile(video_path)
    shutil.rmtree(seq_out_dir)
