import os
import json
import jsonlines
import h5py
import networkx as nx
import math
import numpy as np
import base64
import random

class ImageFeaturesDB(object):
    def __init__(self, img_ft_file, image_feat_size):
        self.image_feat_size = image_feat_size
        self.img_ft_file = img_ft_file
        self._feature_store = {}
        self._feature_store_aug_train = {}
        self.vps = {}
        self._feature_store_aug_rand = {}

    def get_image_feature(self, scan, viewpoint):
        """
        获取图像特征
        Args:
            scan: 扫描ID
            viewpoint: 视点ID
        Returns:
            特征数组 shape: (36, image_feat_size)
        """
        key = f'{scan}_{viewpoint}'
        # 2.1 先检查缓存
        if key in self._feature_store:
            return self._feature_store[key]
        # 2.2 缓存中没有，根据文件类型加载
        if self.img_ft_file.endswith('.tsv'):
            # TSV文件应该在初始化时已经预加载到_feature_store中
            # 如果走到这里说明该viewpoint不存在于TSV文件中
            print(f"Warning: Feature not found for {key} in preloaded TSV data")
            raise KeyError(f"Feature not found for scan={scan}, viewpoint={viewpoint}")
        elif self.img_ft_file.endswith('.hdf5'):
            # HDF5文件按需加载
            try:
                with h5py.File(self.img_ft_file, 'r') as f:
                    if key in f:
                        # 从HDF5读取特征
                        raw_features = f[key][...]
                        # 处理特征维度
                        if raw_features.shape[1] >= self.image_feat_size:
                            # 如果特征维度大于等于需要的维度，截取前image_feat_size维
                            ft = raw_features[:, :self.image_feat_size].astype(np.float32)
                        else:
                            # 如果特征维度小于需要的维度，进行填充
                            ft = np.zeros((36, self.image_feat_size), dtype=np.float32)
                            ft[:, :raw_features.shape[1]] = raw_features.astype(np.float32)
                            print(
                                f"Warning: Padded features for {key} from {raw_features.shape[1]} to {self.image_feat_size}")
                        # 缓存特征
                        self._feature_store[key] = ft
                        return ft
                    else:
                        print(f"Warning: Key {key} not found in HDF5 file")
                        raise KeyError(f"Feature not found for scan={scan}, viewpoint={viewpoint}")
            except Exception as e:
                print(f"Error loading features from HDF5: {e}")
                raise
        else:
            raise ValueError(f"Unsupported file format: {self.img_ft_file}. Supported formats: .tsv, .hdf5")


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

def new_simulator(connectivity_dir, scan_data_dir=None):
    import MatterSim

    # Simulator image parameters
    WIDTH = 640
    HEIGHT = 480
    VFOV = 60

    sim = MatterSim.Simulator()
    if scan_data_dir:
        sim.setDatasetPath(scan_data_dir)
    sim.setNavGraphPath(connectivity_dir)
    sim.setRenderingEnabled(False)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.setBatchSize(1)
    sim.initialize()

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
        if ix == 0:
            sim.newEpisode(['ZMojNkEp431'], ['2f4d90acd4024c269fb0efe49a8ac540'], [0], [math.radians(-30)])
        elif ix % 12 == 0:
            sim.makeAction([0], [1.0], [1.0])
        else:
            sim.makeAction([0], [1.0], [0])

        state = sim.getState()[0]
        assert state.viewIndex == ix

        heading = state.heading - base_heading
        elevation = state.elevation - base_elevation

        feature[ix, :] = angle_feature(heading, elevation, angle_feat_size)
    return feature

def get_all_point_angle_feature(sim, angle_feat_size):
    return [get_point_angle_feature(sim, angle_feat_size, baseViewId) for baseViewId in range(36)]
