import gc
import json
import jsonlines
import numpy as np
import math
import time
import lmdb
import msgpack
import msgpack_numpy
msgpack_numpy.patch()
import h5py
from tqdm import tqdm
from .common import load_nav_graphs
from .common import get_angle_fts, get_view_rel_angles
from .common import calculate_vp_rel_pos_fts
from .common import softmax
MAX_DIST = 30   # normalize
MAX_STEP = 10   # normalize
TRAIN_MAX_STEP = 20

class ReverieTextPathData(object):
    def __init__(
        self, anno_files, img_ft_file, obj_ft_file, scanvp_cands_file, connectivity_dir,
        image_feat_size=2048, image_prob_size=1000, angle_feat_size=4,
        obj_feat_size=None, obj_prob_size=None, max_objects=20,
        max_txt_len=100, in_memory=True, act_visited_node=False,
        obj_image_h=480, obj_image_w=640, scan_ranges=None,
    ):

        # 支持单个文件或文件列表
        if isinstance(img_ft_file, str):
            self.img_ft_file = [img_ft_file]
        elif isinstance(img_ft_file, list):
            self.img_ft_file = img_ft_file
        else:
            raise ValueError("img_ft_file must be a string or a list of strings")
        self.obj_ft_file = obj_ft_file

        self.image_feat_size = image_feat_size
        self.image_prob_size = image_prob_size
        self.angle_feat_size = angle_feat_size
        self.obj_feat_size = obj_feat_size
        self.obj_prob_size = obj_prob_size

        self.obj_image_h = obj_image_h
        self.obj_image_w = obj_image_w
        self.obj_image_size = obj_image_h * obj_image_w

        self.max_txt_len = max_txt_len
        self.max_objects = max_objects
        self.act_visited_node = act_visited_node

        if scan_ranges is not None:
            self.scan_idxs = set([i for i in range(scan_ranges[0], scan_ranges[1])])
        else:
            self.scan_idxs = None
        self.scanvp_cands = dict()
        for scanvp_cand_file in scanvp_cands_file:
            self.scanvp_cands.update(json.load(open(scanvp_cand_file)))
        self.graphs, self.shortest_distances, self.shortest_paths = load_nav_graphs(connectivity_dir)
        self.all_point_rel_angles = [get_view_rel_angles(baseViewId=i) for i in range(36)]
        self.all_point_angle_fts = [get_angle_fts(x[:, 0], x[:, 1], self.angle_feat_size) for x in self.all_point_rel_angles]

        self.data = []
        exist_scanvps_hm3d = set()
        exist_scanvps_reverie = set()
        exist_scanvps = set()
        for data_type, anno_file in enumerate(anno_files):
            with jsonlines.open(anno_file, 'r') as f:
                for item in f:
                    if self.scan_idxs is None or int(item['scan'].split('-')[0]) in self.scan_idxs:
                        item["data"] = "hm3d" if data_type == 0 else "reverie"
                        self.data.append(item)
                        for vp in item['path'] + item['pos_vps']:
                            exist_scanvps.add('%s_%s'%(item['scan'], vp))

        print("Augmentation data:", len(self.data))

        # 处理多个图像特征文件 - 关键优化点
        self.img_env = []
        self.is_hdf5 = []
        self.img_h5_handles = []  # 保存HDF5文件句柄

        for img_ft_file in self.img_ft_file:
            if ".hdf5" in img_ft_file:
                self.is_hdf5.append(True)
                # 关键修改：使用 'r' 模式并设置 libver='latest' 提高兼容性
                # swmr=True 允许多进程同时读取
                try:
                    print(f"Opening HDF5 file: {img_ft_file}")
                    h5_file = h5py.File(img_ft_file, 'r', libver='latest', swmr=True)
                    self.img_h5_handles.append(h5_file)
                    self.img_env.append(img_ft_file)  # 保存路径用于标识
                    print(f"Successfully opened HDF5 file with {len(h5_file.keys())} keys")
                except Exception as e:
                    print(f"ERROR opening HDF5 file {img_ft_file}: {e}")
                    print(f"Will try opening without SWMR mode...")
                    try:
                        h5_file = h5py.File(img_ft_file, 'r')
                        self.img_h5_handles.append(h5_file)
                        self.img_env.append(img_ft_file)
                        print(f"Successfully opened without SWMR")
                    except Exception as e2:
                        print(f"FATAL: Cannot open HDF5 file: {e2}")
                        raise
            else:
                self.is_hdf5.append(False)
                self.img_h5_handles.append(None)
                self.img_env.append(lmdb.open(img_ft_file, readonly=True))

        # 打开对象特征LMDB
        try:
            self.obj_env = lmdb.open(self.obj_ft_file, readonly=True)
            print(f"Successfully opened object LMDB")
        except Exception as e:
            print(f"ERROR opening object LMDB: {e}")
            raise

        # 预加载特征到内存
        self.in_memory = in_memory
        if self.in_memory:
            st_time = time.time()
            print(f'Loading {len(exist_scanvps)} scanvps into memory...')
            self._feature_store = {}

            # 添加进度条和错误计数
            exist_scanvps_list = sorted(list(exist_scanvps))  # 排序以便调试
            failed_count = 0
            self._load_count = 0

            # 使用tqdm显示进度
            pbar = tqdm(exist_scanvps_list,
                        desc=f'Loading features')  # 只在主进程显示进度条

            for scanvp in pbar:
                try:
                    scan, vp = scanvp.split('_')
                    self.get_scanvp_feature(scan, vp)
                    self._load_count += 1

                    # 每1000个特征更新一次进度信息
                    if self._load_count % 1000 == 0:
                        pbar.set_postfix({
                            'loaded': self._load_count,
                            'failed': failed_count,
                            'cache_size': len(self._feature_store)
                        })
                        gc.collect()  # 定期垃圾回收

                except KeyError as e:
                    failed_count += 1
                    if failed_count <= 10:  # 只打印前10个错误
                        print(f"\nWARNING: Feature not found for {scanvp}: {e}")
                    continue
                except Exception as e:
                    failed_count += 1
                    if failed_count <= 10:
                        print(f"\nERROR loading {scanvp}: {e}")
                    continue

            pbar.close()
            elapsed_time = (time.time() - st_time) / 60.0
            print(f'Preloaded {len(self._feature_store)} scanvps '
                  f'in {elapsed_time:.2f} min ({failed_count} failed)')

            # 最终垃圾回收
            gc.collect()

    def __len__(self):
        return len(self.data)

    def __del__(self):
        # 关闭所有HDF5和LMDB环境
        for h5_handle in self.img_h5_handles:
            if h5_handle is not None:
                try:
                    h5_handle.close()
                except:
                    pass

        for img_env, is_hdf5 in zip(self.img_env, self.is_hdf5):
            if not is_hdf5:  # 只有LMDB需要关闭
                try:
                    img_env.close()
                except:
                    pass

        try:
            self.obj_env.close()
        except:
            pass

    def get_scanvp_feature(self, scan, viewpoint):
        """优化后的特征获取方法"""
        key = '%s_%s' % (scan, viewpoint)
        # 如果已经在缓存中，直接返回
        if self.in_memory and key in self._feature_store:
            view_fts, obj_fts, obj_attrs = self._feature_store[key]
        else:
            view_fts = None
            # 遍历所有图像特征文件，找到第一个包含该key的文件
            for idx, (is_hdf5, h5_handle) in enumerate(zip(self.is_hdf5, self.img_h5_handles)):
                try:
                    if is_hdf5:
                        # 使用已打开的HDF5句柄
                        if h5_handle is not None and key in h5_handle:
                            # 直接从HDF5读取，使用[...]强制加载到内存
                            view_fts = h5_handle[key][...].astype(np.float32)
                            break
                    else:
                        # LMDB读取
                        img_env = self.img_env[idx]
                        with img_env.begin() as txn:
                            data = txn.get(key.encode('ascii'))
                            if data is not None:
                                view_fts = msgpack.unpackb(data)
                                break
                except Exception as e:
                    # 如果某个文件读取失败，继续尝试下一个
                    if self._load_count <= 10 or self._load_count % 1000 == 0:
                        print(f"\nWarning: Error reading {key} from file {idx}: {e}")
                    continue
            # 如果所有文件都没找到该特征
            if view_fts is None:
                raise KeyError(f"Feature key '{key}' not found in any of the image feature files")
            # 截取所需的特征维度
            view_fts = view_fts[:, :self.image_feat_size + self.image_prob_size]
            # 读取对象特征
            obj_attrs = {}
            obj_fts = np.zeros((0, self.obj_feat_size + self.obj_prob_size), dtype=np.float32)
            try:
                with self.obj_env.begin() as txn:
                    d = txn.get(key.encode('ascii'))
                if d is not None:
                    d = msgpack.unpackb(d)
                    obj_fts = d['fts'][:self.max_objects, :self.obj_feat_size + self.obj_prob_size]
                    for attr_key, attr_value in d.items():
                        if attr_key == 'fts':
                            continue
                        if attr_key in ['directions', 'sizes', 'bboxes', 'obj_ids']:
                            obj_attrs[attr_key] = attr_value[:self.max_objects]
                        if attr_key == 'centers':
                            obj_attrs['directions'] = np.array(attr_value).astype(np.float32)
                        if attr_key == 'bboxes':
                            obj_attrs['sizes'] = np.array(obj_attrs['bboxes']).astype(np.float32)[:, 2:]
            except Exception as e:
                # 对象特征可能不存在，这是正常的
                pass
            # 如果启用内存缓存，保存到缓存中
            if self.in_memory:
                self._feature_store[key] = (view_fts, obj_fts, obj_attrs)
        # 定期垃圾回收（每1000次）
        if hasattr(self, '_load_count') and self._load_count % 1000 == 0:
            gc.collect()
        return view_fts, obj_fts, obj_attrs

    def get_obj_label(self, item, last_vp_objids):
        gt_obj_id = item['instr_id'].split('_')[1]
        for k, obj_id in enumerate(last_vp_objids):
            if obj_id == gt_obj_id:
                obj_label = k
                break
        else:
            obj_label = -100 # ignore
        return obj_label

    def get_act_labels(self, end_vp, item, gmap_vpids, gmap_visited_masks, traj_cand_vpids):
        scan = item['scan']
        pos_vps = item['pos_vps']
        if end_vp in pos_vps:
            global_act_label = local_act_label = 0
        else:
            global_act_label = local_act_label = -100
            cand_min_dist = float('inf')
            for k, cand_vp in enumerate(gmap_vpids):
                if (k > 0) and (not gmap_visited_masks[k]):
                    min_dist = min([self.shortest_distances[scan][end_vp][cand_vp] \
                        + self.shortest_distances[scan][cand_vp][pos_vp] for pos_vp in pos_vps])
                    if min_dist < cand_min_dist:
                        cand_min_dist = min_dist
                        global_act_label = k # [stop] is 0
            cand_min_dist = float('inf')
            for k, cand_vp in enumerate(traj_cand_vpids[-1]):
                min_dist = min([self.shortest_distances[scan][end_vp][cand_vp] \
                    + self.shortest_distances[scan][cand_vp][pos_vp] for pos_vp in pos_vps])
                if min_dist < cand_min_dist:
                    cand_min_dist = min_dist
                    local_act_label = k + 1 # [stop] is 0
        return global_act_label, local_act_label

    def get_input(
        self, idx, end_vp_type, return_img_probs=False, return_act_label=False, 
        return_obj_label=False, end_vp=None
    ):
        item = self.data[idx]
        scan = item['scan']
        start_vp = item['path'][0]
        start_heading = item.get('heading', 0)
        pos_vps = item['pos_vps']
        gt_path = item['path']

        if end_vp is None:
            if end_vp_type == 'pos':
                end_vp = pos_vps[np.random.randint(len(pos_vps))]
            elif end_vp_type == 'neg_in_gt_path':
                end_vps = [vp for vp in gt_path if vp not in pos_vps]
                if len(end_vps) == 0:
                    end_vps = gt_path
                end_vp = end_vps[np.random.randint(len(end_vps))]
            elif end_vp_type == 'neg_others':
                noneg_vp_set = set(pos_vps + gt_path)
                end_vps = [vp for vp in self.shortest_paths[scan][start_vp].keys() if vp not in noneg_vp_set]
                if len(end_vps) == 0:
                    end_vps = gt_path[:-1]
                end_vp = end_vps[np.random.randint(len(end_vps))]

        gt_path = self.shortest_paths[scan][start_vp][end_vp]
        cur_heading, cur_elevation = self.get_cur_angle(scan, gt_path, start_heading)

        if len(gt_path) > TRAIN_MAX_STEP:
            # truncate trajectory
            gt_path = gt_path[:TRAIN_MAX_STEP] + [end_vp]
            
        traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids, \
            last_vp_angles, last_vp_objids = self.get_traj_pano_fts(scan, gt_path)

        # global: the first token is [stop]
        gmap_vpids, gmap_step_ids, gmap_visited_masks, gmap_pos_fts, gmap_pair_dists = \
            self.get_gmap_inputs(scan, gt_path, cur_heading, cur_elevation)

        # local: the first token is [stop]
        vp_pos_fts = self.get_vp_pos_fts(scan, start_vp, end_vp,
            traj_cand_vpids[-1], cur_heading, cur_elevation, len(traj_nav_types[-1]))

        outs = {
            'instr_id': item['instr_id'],
            'instr_encoding': item['instr_encoding'][:self.max_txt_len],
            
            'traj_view_img_fts': [x[:, :self.image_feat_size] for x in traj_view_img_fts],
            'traj_obj_img_fts': [x[:, :self.obj_feat_size] for x in traj_obj_img_fts],
            'traj_loc_fts': traj_loc_fts,
            'traj_nav_types': traj_nav_types,
            'traj_cand_vpids': traj_cand_vpids,
            'traj_vpids': gt_path,

            'gmap_vpids': gmap_vpids,
            'gmap_step_ids': gmap_step_ids,
            'gmap_visited_masks': gmap_visited_masks,
            'gmap_pos_fts': gmap_pos_fts,
            'gmap_pair_dists': gmap_pair_dists,

            'vp_pos_fts': vp_pos_fts,
            # 'vp_objids': last_vp_objids,
            'vp_angles': last_vp_angles,
        }

        if return_obj_label:
            outs['obj_labels'] = self.get_obj_label(item, last_vp_objids)

        if return_act_label:
            global_act_label, local_act_label = self.get_act_labels(
                end_vp, item, gmap_vpids, gmap_visited_masks, traj_cand_vpids
            )
            outs['global_act_labels'] = global_act_label
            outs['local_act_labels'] = local_act_label

        if return_img_probs:
            # TODO: whether adding gmap img probs
            outs['vp_view_probs'] = softmax(traj_view_img_fts[-1][:, self.image_feat_size:], dim=1)
            outs['vp_obj_probs'] = softmax(traj_obj_img_fts[-1][:, self.obj_feat_size:], dim=1)

        return outs

    def get_cur_angle(self, scan, path, start_heading):
        if len(path) < 2:
            heading = start_heading
            elevation = 0
        else:
            prev_vp = path[-2]
            cur_vp = path[-1]
            viewidx = self.scanvp_cands['%s_%s'%(scan, prev_vp)][cur_vp][0]
            heading = (viewidx % 12) * math.radians(30)
            elevation = (viewidx // 12 - 1) * math.radians(30)
        return heading, elevation

    def get_traj_pano_fts(self, scan, path):
        '''
        Tokens in each pano: [cand_views, noncand_views, objs]
        Each token consists of (img_fts, loc_fts (ang_fts, box_fts), nav_types)
        '''
        traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids = [], [], [], [], []

        for vp in path:
            view_fts, obj_img_fts, obj_attrs = self.get_scanvp_feature(scan, vp)

            view_img_fts, view_angles, cand_vpids = [], [], []
            # cand views
            nav_cands = self.scanvp_cands['%s_%s'%(scan, vp)]
            used_viewidxs = set()
            for k, v in nav_cands.items():
                used_viewidxs.add(v[0])
                view_img_fts.append(view_fts[v[0]])
                # TODO: whether using correct heading at each step
                view_angle = self.all_point_rel_angles[12][v[0]]
                view_angles.append([view_angle[0] + v[2], view_angle[1] + v[3]])
                cand_vpids.append(k)
            # non cand views
            view_img_fts.extend([view_fts[idx] for idx in range(36) if idx not in used_viewidxs])
            view_angles.extend([self.all_point_rel_angles[12][idx] for idx in range(36) if idx not in used_viewidxs])
            # combine cand views and noncand views
            view_img_fts = np.stack(view_img_fts, 0)    # (n_views, dim_ft)
            view_angles = np.stack(view_angles, 0)
            view_ang_fts = get_angle_fts(view_angles[:, 0], view_angles[:, 1], self.angle_feat_size)
            view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)
            
            # object features
            num_objs = obj_img_fts.shape[0]
            obj_angles = np.zeros((num_objs, 2), dtype=np.float32)
            obj_ang_fts = np.zeros((num_objs, self.angle_feat_size), dtype=np.float32)
            obj_box_fts = np.zeros((num_objs, 3), dtype=np.float32)
            if num_objs > 0:
                for k, (w, h) in enumerate(obj_attrs['sizes']):
                    obj_angles[k] = obj_attrs['directions'][k]
                    obj_box_fts[k] = [h/self.obj_image_h, w/self.obj_image_w, (h*w)/self.obj_image_size]           
                obj_ang_fts = get_angle_fts(obj_angles[:, 0], obj_angles[:, 1], self.angle_feat_size)

            # combine pano features
            traj_view_img_fts.append(view_img_fts)
            traj_obj_img_fts.append(obj_img_fts)
            traj_loc_fts.append(
                np.concatenate(
                    [np.concatenate([view_ang_fts, view_box_fts], 1),
                     np.concatenate([obj_ang_fts, obj_box_fts], 1)], axis=0
                )
            )
            traj_nav_types.append(
                [1] * len(cand_vpids) + [0] * (36 - len(used_viewidxs)) + [2] * len(obj_img_fts)
            )
            traj_cand_vpids.append(cand_vpids)

            last_vp_objids = obj_attrs.get('obj_ids', [])
            last_vp_angles = np.concatenate([view_angles, obj_angles], 0)

        return traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids, \
               last_vp_angles, last_vp_objids
        
    def get_gmap_inputs(self, scan, path, cur_heading, cur_elevation):
        scan_graph = self.graphs[scan]
        cur_vp = path[-1]

        visited_vpids, unvisited_vpids = {}, {}
        for t, vp in enumerate(path):
            visited_vpids[vp] = t + 1
            if vp in unvisited_vpids:
                del unvisited_vpids[vp]
            for next_vp in self.scanvp_cands['%s_%s'%(scan, vp)].keys():
                if next_vp not in visited_vpids:
                    unvisited_vpids[next_vp] = 0
        # add [stop] token
        gmap_vpids = [None] + list(visited_vpids.keys()) + list(unvisited_vpids.keys())
        gmap_step_ids = [0] + list(visited_vpids.values()) + list(unvisited_vpids.values())
        if self.act_visited_node:
            gmap_visited_masks = [0]
            for vp in gmap_vpids[1:]:
                if vp == path[-1]:
                    gmap_visited_masks.append(1)
                else:
                    gmap_visited_masks.append(0)
        else:
            gmap_visited_masks = [0] + [1] * len(visited_vpids) + [0] * len(unvisited_vpids)

        gmap_pos_fts = self.get_gmap_pos_fts(scan, cur_vp, gmap_vpids, cur_heading, cur_elevation)
        
        gmap_pair_dists = np.zeros((len(gmap_vpids), len(gmap_vpids)), dtype=np.float32)
        for i in range(1, len(gmap_vpids)):
            for j in range(i+1, len(gmap_vpids)):
                gmap_pair_dists[i, j] = gmap_pair_dists[j, i] = \
                    self.shortest_distances[scan][gmap_vpids[i]][gmap_vpids[j]]

        return gmap_vpids, gmap_step_ids, gmap_visited_masks, gmap_pos_fts, gmap_pair_dists
    
    def get_gmap_pos_fts(self, scan, cur_vp, gmap_vpids, cur_heading, cur_elevation):
        rel_angles, rel_dists = [], []
        for vp in gmap_vpids:
            if vp is None:
                rel_angles.append([0, 0])
                rel_dists.append([0, 0, 0])
            else:
                rel_heading, rel_elevation, rel_dist = calculate_vp_rel_pos_fts(
                    self.graphs[scan].nodes[cur_vp]['position'], 
                    self.graphs[scan].nodes[vp]['position'],
                    base_heading=cur_heading, base_elevation=cur_elevation,
                )
                rel_angles.append([rel_heading, rel_elevation])
                rel_dists.append(
                    [rel_dist / MAX_DIST, self.shortest_distances[scan][cur_vp][vp] / MAX_DIST, \
                    (len(self.shortest_paths[scan][cur_vp][vp]) - 1) / MAX_STEP]
                )
        rel_angles = np.array(rel_angles).astype(np.float32)
        rel_dists = np.array(rel_dists).astype(np.float32)
        rel_ang_fts = get_angle_fts(rel_angles[:, 0], rel_angles[:, 1], self.angle_feat_size)
        return np.concatenate([rel_ang_fts, rel_dists], 1)
        
    def get_vp_pos_fts(self, scan, start_vp, cur_vp, cand_vpids, cur_heading, cur_elevation, vp_ft_len):
        cur_cand_pos_fts = self.get_gmap_pos_fts(scan, cur_vp, cand_vpids, cur_heading, cur_elevation)
        cur_start_pos_fts = self.get_gmap_pos_fts(scan, cur_vp, [start_vp], cur_heading, cur_elevation)
                
        # add [stop] token at beginning
        vp_pos_fts = np.zeros((vp_ft_len+1, 14), dtype=np.float32)
        vp_pos_fts[:, :7] = cur_start_pos_fts
        vp_pos_fts[1:len(cur_cand_pos_fts)+1, 7:] = cur_cand_pos_fts

        return vp_pos_fts
       
