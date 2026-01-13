import numpy as np
import os
import glob
import pickle

class Node:
    def __init__(self):
        self.id = None
        self.seq_id = None
        self.cam_id = None
        self.track_keys = []  # List of (seq_id, cam_id, obj_id)
        self.reid_average = None
        self.clip_average = None

    def compute_averages(self, all_features):
        reid_vecs = []
        clip_vecs = []

        for key in self.track_keys:
            feat = all_features[key]
            reid_vecs.append(feat['reid'])
            clip_vecs.append(feat['clip'])

        self.reid_average = np.mean(reid_vecs, axis=0)
        self.reid_average /= np.linalg.norm(self.reid_average)

        self.clip_average = np.mean(clip_vecs, axis=0)
        self.clip_average /= np.linalg.norm(self.clip_average)
    
    def save(self):
        pass


def main():
    clusters_dir = 'new_clusters'
    seqs = sorted(glob.glob(os.path.join(clusters_dir, 'seq_*/')))
   
    cameras = ['camera_1', 'camera_2', 'camera_3', 'camera_4', 'camera_5', 'camera_6', 'camera_7']

    nodes = []
    node_id = 0

    for seq in seqs:
        seq_name = os.path.basename(os.path.normpath(seq))
        print(f"Processing {seq_name}")
        for cam in cameras:
            print(f"Processing {cam}")
            cluster_file = f'{clusters_dir}/results/{seq_name}_{cam}_results.pkl'
            # print(f"Loading clusters from {cluster_file}...")
            if not os.path.exists(cluster_file):
                print(f"Cluster file not found: {cluster_file}")
                continue

            with open(cluster_file, "rb") as f:
                data = pickle.load(f)
            print(f"File: {cluster_file}")


            feature_file_path = os.path.join('data','new_feature_objects', seq_name, f"{seq_name}_{cam}.pkl")
            if not os.path.exists(feature_file_path):
                print(f"Feature file not found: {feature_file_path}")
                continue
            with open(feature_file_path, "rb") as f:
                all_features = pickle.load(f)
            # print(f"Loaded features from {feature_file_path}")


            for i, object in enumerate(data):
                node = Node()
                print(i, object)
                
                node.track_keys = object
                node.id = node_id
                node_id += 1
                node.seq_id = int(seq_name.split('_')[-1])
                node.cam_id = int(cam.split('_')[-1])
                node.compute_averages(all_features)
                # print(node.reid_average.shape, node.clip_average.shape)
                nodes.append(node)
                # print(len(nodes))
                
    # save into 2 files:
    

if __name__ == "__main__":
    main()