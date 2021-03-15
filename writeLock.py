#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 10:49:00 2021

@author: charlie
"""
import torch
import ray
from filelock import FileLock


@ray.remote
def save_with_lock(agent, path):
    lock = FileLock(path)
    with lock:
        torch.save(
                agent.model.state_dict(), path)