''' Batched REVERIE navigation environment '''

import json
import os
import numpy as np
import math
import random
import networkx as nx
import copy
import h5py
import jsonlines
import collections

from utils.data import load_nav_graphs, new_simulator
from utils.data import angle_feature, get_all_point_angle_feature

from .env import EnvBatch, ReverieObjectNavBatch
from utils.data import ImageFeaturesDB
from .data_utils import ObjectFeatureDB

def construct_instrs(instr_files, max_instr_len=512):
    data = []
    for instr_file in instr_files:
        with jsonlines.open(instr_file) as f:
            for item in f:
                newitem = {
                    'instr_id': item['instr_id'], 
                    'objId': item['objid'],
                    'scan': item['scan'],
                    'path': item['path'],
                    'end_vps': item['pos_vps'],
                    'instruction': item['instruction'],
                    'instr_encoding': item['instr_encoding'][:max_instr_len],
                    'heading': np.random.rand() * np.pi * 2,
                }
                data.append(newitem)
    return data


class HM3DReverieObjectNavBatch(ReverieObjectNavBatch):
    ''' Implements the REVERIE navigation task, using discretized viewpoints and pretrained features '''

    def __init__(self, view_db, obj_db, instr_files, connectivity_dir, multi_endpoints=False, multi_startpoints=False,
                 batch_size=64, angle_feat_size=4, max_objects=None, seed=0, name=None, sel_data_idxs=None,
                 scan_ranges=None):

        # 先解析数据
        instr_data = construct_instrs(instr_files, max_instr_len=100)

        # 根据 scan_ranges 过滤数据
        if scan_ranges is not None:
            scan_idxs = set(list(range(scan_ranges[0], scan_ranges[1])))
            new_instr_data = []
            for item in instr_data:
                if int(item['scan'].split('-')[0]) in scan_idxs:
                    new_instr_data.append(item)
            instr_data = new_instr_data

        # 构建 obj2vps
        obj2vps = collections.defaultdict(list)
        for item in instr_data:
            obj2vps['%s_%s' % (item['scan'], item['objId'])].extend(item['end_vps'])

        # 现在调用父类初始化，传入解析后的数据
        super().__init__(
            view_db, obj_db, instr_data, connectivity_dir, obj2vps,
            multi_endpoints=multi_endpoints, multi_startpoints=multi_startpoints,
            batch_size=batch_size, angle_feat_size=angle_feat_size,
            max_objects=max_objects, seed=seed, name=name, sel_data_idxs=sel_data_idxs
        )

        # 如果需要覆盖父类的某些属性，在这里设置
        # 注意：父类已经设置了 self.data, self.scans, self.obj2vps 等，
        # 通常不需要再重复设置

        print('%s loaded with %d instructions, using splits: %s' % (
            self.__class__.__name__, len(self.data), self.name))
