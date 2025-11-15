import os
import json
import networkx as nx
import math
import numpy as np

import lmdb
import msgpack
import msgpack_numpy
msgpack_numpy.patch()
import h5py
import random
from tqdm import tqdm


class ImageFeaturesDB(object):
    """升级版的图像特征数据库,支持单文件或多文件列表"""
    def __init__(self, img_ft_file, image_feat_size):
        """
        Args:
            img_ft_file: 单个文件路径(str)或文件列表(list)
            image_feat_size: 图像特征维度
        """
        self.image_feat_size = image_feat_size
        self._feature_store = {}

        # 支持单文件或文件列表
        if isinstance(img_ft_file, str):
            self.img_ft_file = [img_ft_file]
        elif isinstance(img_ft_file, list):
            self.img_ft_file = img_ft_file
        else:
            raise ValueError("img_ft_file must be a string or a list of strings")
        # 初始化存储结构
        self.img_env = []
        self.is_hdf5 = []
        self.img_h5_handles = []
        print(f"Initializing ImageFeaturesDB with {len(self.img_ft_file)} file(s)...")
        # 使用进度条打开所有文件
        for img_ft_file_path in tqdm(self.img_ft_file, desc="Opening feature files"):
            if ".hdf5" in img_ft_file_path:
                self.is_hdf5.append(True)
                try:
                    print(f"  Opening HDF5 file: {os.path.basename(img_ft_file_path)}")
                    h5_file = h5py.File(img_ft_file_path, 'r', libver='latest', swmr=True)
                    self.img_h5_handles.append(h5_file)
                    self.img_env.append(img_ft_file_path)
                    print(f"  Successfully opened HDF5 file with {len(h5_file.keys())} keys")
                except Exception as e:
                    print(f"  ERROR opening HDF5 file with SWMR: {e}")
                    print(f"  Trying without SWMR mode...")
                    try:
                        h5_file = h5py.File(img_ft_file_path, 'r')
                        self.img_h5_handles.append(h5_file)
                        self.img_env.append(img_ft_file_path)
                        print(f"  Successfully opened without SWMR")
                    except Exception as e2:
                        print(f"  FATAL: Cannot open HDF5 file: {e2}")
                        raise
            else:
                self.is_hdf5.append(False)
                self.img_h5_handles.append(None)
                print(f"  Opening LMDB file: {os.path.basename(img_ft_file_path)}")
                self.img_env.append(lmdb.open(img_ft_file_path, readonly=True))
                print(f"  Successfully opened LMDB file")

        print(f"ImageFeaturesDB initialized successfully!")

    def __del__(self):
        """清理资源"""
        for h5_handle in self.img_h5_handles:
            if h5_handle is not None:
                try:
                    h5_handle.close()
                except:
                    pass
        for img_env, is_hdf5 in zip(self.img_env, self.is_hdf5):
            if not is_hdf5:
                try:
                    img_env.close()
                except:
                    pass

    def get_image_feature(self, scan, viewpoint):
        """获取图像特征，支持从多个文件中查找"""
        key = '%s_%s' % (scan, viewpoint)
        # 先检查缓存
        if key in self._feature_store:
            return self._feature_store[key]
        # 依次尝试从各个文件读取
        ft = None
        for idx, (is_hdf5, h5_handle) in enumerate(zip(self.is_hdf5, self.img_h5_handles)):
            try:
                if is_hdf5:
                    # HDF5读取
                    if h5_handle is not None and key in h5_handle:
                        ft = h5_handle[key][...].astype(np.float32)
                        break
                else:
                    # LMDB读取
                    img_env = self.img_env[idx]
                    with img_env.begin() as txn:
                        data = txn.get(key.encode('ascii'))
                        if data is not None:
                            ft = msgpack.unpackb(data)
                            break
            except Exception as e:
                # 如果某个文件读取失败，继续尝试下一个
                continue
        # 如果所有文件都没找到该特征
        if ft is None:
            raise KeyError(f"Feature key '{key}' not found in any of the image feature files")
        # 截取所需的特征维度
        ft = ft[:, :self.image_feat_size].astype(np.float32)
        # 缓存特征
        self._feature_store[key] = ft
        return ft

class ImageFeaturesDB2(object):
    def __init__(self, img_ft_files, image_feat_size):
        self.image_feat_size = image_feat_size
        self.img_ft_file = img_ft_files
        self._feature_stores = {}
        for name in img_ft_files:
            self._feature_stores[name] = {}
            with h5py.File(name, 'r') as f:
                for key in f.keys():
                    ft = f[key][...][:, :self.image_feat_size].astype(np.float32)
                    self._feature_stores[name][key] = ft 
        self.env_names = list(self._feature_stores.keys())
        print(self.env_names)
        

    def get_image_feature(self, scan, viewpoint):
        key = '%s_%s' % (scan, viewpoint)
        env_name = random.choice(self.env_names)
        if key in self._feature_stores[env_name]:
            ft = self._feature_stores[env_name][key]
        else:
            with h5py.File(env_name, 'r') as f:
                ft = f[key][...][:, :self.image_feat_size].astype(np.float32)
                self._feature_stores[env_name][key] = ft
        return ft

def load_nav_graphs(connectivity_dir, scans):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    graphs = {}
    for scan in scans:
        with open(os.path.join(connectivity_dir, '%s_connectivity.json' % scan)) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i,item in enumerate(data):
                if item['included']:
                    for j,conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                    item['pose'][7], item['pose'][11]]);
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs

def new_simulator(connectivity_dir, scan_data_dir=None, width=640, height=480, vfov=60):
    import MatterSim

    sim = MatterSim.Simulator()
    if scan_data_dir:
        sim.setDatasetPath(scan_data_dir)
    sim.setNavGraphPath(connectivity_dir)
    sim.setRenderingEnabled(False)
    sim.setCameraResolution(width, height)
    sim.setCameraVFOV(math.radians(vfov))
    sim.setDiscretizedViewingAngles(True)
    sim.setBatchSize(1)
    sim.initialize()
    #sim.init()

    return sim

def angle_feature(heading, elevation, angle_feat_size):
    return np.array(
        [math.sin(heading), math.cos(heading), math.sin(elevation), math.cos(elevation)] * (angle_feat_size // 4),
        dtype=np.float32)

def get_point_angle_feature(sim, angle_feat_size, baseViewId=0):
    feature = np.empty((36, angle_feat_size), np.float32)
    base_heading = (baseViewId % 12) * math.radians(30)
    base_elevation = (baseViewId // 12 - 1) * math.radians(30)

    for ix in range(36):
        # if ix == 0:
        #     sim.newEpisode(['ZMojNkEp431'], ['2f4d90acd4024c269fb0efe49a8ac540'], [0], [math.radians(-30)])
        # elif ix % 12 == 0:
        #     sim.makeAction([0], [1.0], [1.0])
        # else:
        #     sim.makeAction([0], [1.0], [0])

        # state = sim.getState()[0]
        # assert state.viewIndex == ix

        # heading = state.heading - base_heading
        # elevation = state.elevation - base_elevation
        heading = (ix % 12) * math.radians(30) - base_heading
        elevation = (ix // 12 - 1) * math.radians(30) - base_elevation

        feature[ix, :] = angle_feature(heading, elevation, angle_feat_size)
    return feature

def get_all_point_angle_feature(sim, angle_feat_size):
    return [get_point_angle_feature(sim, angle_feat_size, baseViewId) for baseViewId in range(36)]

