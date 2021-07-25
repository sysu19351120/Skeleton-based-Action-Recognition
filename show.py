# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 13:49:36 2021

@author: nkliu
"""
import torch
import numpy as np
from tool.visualise import visualise
from tool.graph import Graph
import os
import glob
paths=glob.glob('./gen_data/000/*.npy')
for i,sample_path in enumerate(paths):
    #sample_path = './data/train/004/P105S06G10B30H30UC032000LC022000A153R0_09261453.npy'
    sample = np.load(sample_path)
    sample[:,1,:,:,:]=0
    sample=torch.from_numpy(sample)
    visualise(sample, graph=Graph(), is_3d=True)