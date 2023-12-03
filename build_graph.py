import os
import pickle

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.parse_traj import ParseMMTraj
from common.mbr import MBR

def gps2grid(lat, lng, mbr, grid_size):
    """
    mbr:
        MBR class.
    grid size:
        int. in meter
    """
    LAT_PER_METER = 8.993203677616966e-06
    LNG_PER_METER = 1.1700193970443768e-05
    lat_unit = LAT_PER_METER * grid_size
    lng_unit = LNG_PER_METER * grid_size
    
    max_xid = int((mbr.max_lat - mbr.min_lat) / lat_unit) + 1
    max_yid = int((mbr.max_lng - mbr.min_lng) / lng_unit) + 1
    
    locgrid_x = int((lat - mbr.min_lat) / lat_unit) + 1
    locgrid_y = int((lng - mbr.min_lng) / lng_unit) + 1
    if locgrid_x < 0: locgrid_x = 0
    if locgrid_x > max_xid: locgrid_x = max_xid
    if locgrid_y < 0: locgrid_y = 0
    if locgrid_y > max_yid: locgrid_y = max_yid
    return locgrid_x, locgrid_y

def get_length_level_loc(raw_rn_dict, raw_eid, mbr, grid_size):
    raw_eid = str(raw_eid)
    length = raw_rn_dict[raw_eid]['length']
    level = raw_rn_dict[raw_eid]['level']
    coord = raw_rn_dict[raw_eid]['coords']
    start_lat, start_lng = coord[0][0], coord[0][1]
    end_lat, end_lng = coord[-1][0], coord[-1][1]
    start_lat, start_lng = gps2grid(start_lat, start_lng, mbr, grid_size)
    end_lat, end_lng = gps2grid(end_lat, end_lng, mbr, grid_size)
    return start_lat, start_lng, end_lat, end_lng, length, level

def build_global_POI_checkin_graph(dst_dir, new2raw_rid, raw_rn_dict, mbr, grid_size):
    G = nx.DiGraph()
    maxn = 0
    # 第一步，顶点都在new2raw_rid中，先把所有节点加到图中
    for k, v in new2raw_rid.items():
        # print(k+1)
        raw_eid = v
        start_lat, start_lng, end_lat, end_lng, length, level = get_length_level_loc(raw_rn_dict, raw_eid, mbr, grid_size)
        G.add_node(k,
                    start_lat=start_lat,
                    start_lng=start_lng,
                    end_lat=end_lat,
                    end_lng=end_lng,
                    length=length,
                    level=level,
                    freq_cnt=1)
    # exit()
    #添加边
    parser = ParseMMTraj()
    cnt = 0
    for filename in tqdm(os.listdir(dst_dir)):
        #第一步，先解析轨迹
        
        trajs = parser.parse(os.path.join(dst_dir, filename))
        # print("filename:", filename)
        for traj in trajs:
            cnt += 1
            tmp_pt_list = traj.pt_list
            first_flag = 0
            for i in range(len(tmp_pt_list)):
                candi_pt = tmp_pt_list[i].data['candi_pt']
                if candi_pt is not None:   #找到第一个不为空的节点
                    pre_id = str(candi_pt.eid) #记录u的eid
                    first_flag = i    
                    
                    break

            for i in range(first_flag+1, len(tmp_pt_list)):
                candi_pt = tmp_pt_list[i].data['candi_pt']
                if candi_pt is not None:
                    u = str(pre_id)
                    v = str(candi_pt.eid)
                    
                    # Add edges
                    if G.has_edge(u, v):
                        G.edges[u, v]['weight'] += 1
                    else:  # Add new edge
                        G.add_edge(u, v, weight=1)
                    pre_id = v
        # break  
    return G


def save_graph_to_csv(G, dst_dir):
    # Save graph to an adj matrix file and a nodes file
    # Adj matrix file: edge from row_idx to col_idx with weight; Rows and columns are ordered according to nodes file.
    # Nodes file: node_name/poi_id, node features (category, location); Same node order with adj matrix.

    # Save adj matrix
    nodelist = G.nodes()
    A = nx.adjacency_matrix(G, nodelist=nodelist)
    # np.save(os.path.join(dst_dir, 'adj_mtx.npy'), A.todense())
    np.savetxt(os.path.join(dst_dir, 'graph_A.csv'), A.todense(), delimiter=',')

    # Save nodes list
    nodes_data = list(G.nodes.data())  # [(node_name, {attr1, attr2}),...]
    with open(os.path.join(dst_dir, 'graph_X.csv'), 'w') as f:
        print('node_name/road_id,start_lat,start_lng,end_lat,end_lng,length,level, freq_cnt', file=f)
        for each in nodes_data:
            # print(each[1])
            if 'start_lat' not in each[1]:
                print(each)
            node_name = each[0]
            start_lat = each[1]['start_lat']
            start_lng = each[1]['start_lng']
            end_lat = each[1]['end_lat']
            end_lng = each[1]['end_lng']
            length = each[1]['length']
            level = each[1]['level']
            freq_cnt = each[1]['freq_cnt']
            print(f'{node_name},{start_lat},'
                  f'{start_lng},{end_lat},{end_lng},'
                  f'{length},{level},{freq_cnt}', file=f)
            # exit()

def save_graph_to_pickle(G, dst_dir):
    pickle.dump(G, open(os.path.join(dst_dir, 'graph.pkl'), 'wb'))


def save_graph_edgelist(G, dst_dir):
    nodelist = G.nodes()
    node_id2idx = {k: v for v, k in enumerate(nodelist)}

    with open(os.path.join(dst_dir, 'graph_node_id2idx.txt'), 'w') as f:
        for i, node in enumerate(nodelist):
            print(f'{node}, {i}', file=f)

    with open(os.path.join(dst_dir, 'graph_edge.edgelist'), 'w') as f:
        for edge in nx.generate_edgelist(G, data=['weight']):
            src_node, dst_node, weight = edge.split(' ')
            
            print(f'{node_id2idx[src_node]} {node_id2idx[dst_node]} {weight}', file=f)


def load_graph_adj_mtx(path):
    """A.shape: (num_node, num_node), edge from row_index to col_index with weight"""
    A = np.loadtxt(path, delimiter=',')
    A[A!=0.] = 1.
    A[A==0.] = 1e-10
    # A = calculate_laplacian_matrix(A, 'hat_rw_normd_lap_mat')
    A1 = A
    print(A1.shape)
    # A2 = np.matmul(A,A)
    # A3 = np.matmul(A2,A)
    # A2[A2!=0.] = 1.
    # A3[A3!=0.] = 1.
    # print(np.sum(A1)/A1.shape[0], np.sum(A2)/A1.shape[0], np.sum(A3)/A1.shape[0])
    
    return A1

 
def load_graph_node_features(path, feature1='start_lat', feature2='start_lng',
                             feature3='end_lat', feature4='end_lng', feature5='length', feature6='level'):
    """X.shape: (num_node, 4), four features: checkin cnt, poi cat, latitude, longitude"""
    df = pd.read_csv(path)
    rlt_df = df[[feature1, feature2, feature3, feature4, feature5, feature6]]
    X = rlt_df.to_numpy()
    length_max = X[:,4].max()
    X[:,4] = X[:,4]/length_max

    from sklearn.preprocessing import OneHotEncoder
    one_hot_encoder = OneHotEncoder()
    cat_list = list(X[:, 5])
    one_hot_encoder.fit(list(map(lambda x: [x], cat_list)))
    one_hot_rlt = one_hot_encoder.transform(list(map(lambda x: [x], cat_list))).toarray()
    num_cats = one_hot_rlt.shape[-1]
    final_x = np.zeros((X.shape[0], X.shape[-1] - 1 + num_cats), dtype=np.float32)
    final_x[:,:5] = X[:,:5]
    final_x[:,5:] = one_hot_rlt
    
    return final_x


def print_graph_statisics(G):
    print(f"Num of nodes: {G.number_of_nodes()}")
    print(f"Num of edges: {G.number_of_edges()}")

    # Node degrees (mean and percentiles)
    node_degrees = [each[1] for each in G.degree]
    print(f"Node degree (mean): {np.mean(node_degrees):.2f}")
    for i in range(0, 101, 20):
        print(f"Node degree ({i} percentile): {np.percentile(node_degrees, i)}")

    # Edge weights (mean and percentiles)
    edge_weights = []
    for n, nbrs in G.adj.items():
        for nbr, attr in nbrs.items():
            weight = attr['weight']
            edge_weights.append(weight)
    print(f"Edge frequency (mean): {np.mean(edge_weights):.2f}")
    for i in range(0, 101, 20):
        print(f"Edge frequency ({i} percentile): {np.percentile(edge_weights, i)}")


def calculate_laplacian_matrix(adj_mat, mat_type):
    n_vertex = adj_mat.shape[0]

    # row sum
    deg_mat_row = np.asmatrix(np.diag(np.sum(adj_mat, axis=1)))
    # column sum
    # deg_mat_col = np.asmatrix(np.diag(np.sum(adj_mat, axis=0)))
    deg_mat = deg_mat_row

    adj_mat = np.asmatrix(adj_mat)
    id_mat = np.asmatrix(np.identity(n_vertex))

    if mat_type == 'com_lap_mat':
        # Combinatorial
        com_lap_mat = deg_mat - adj_mat
        return com_lap_mat
    elif mat_type == 'wid_rw_normd_lap_mat':
        # For ChebConv
        rw_lap_mat = np.matmul(np.linalg.matrix_power(deg_mat, -1), adj_mat)
        rw_normd_lap_mat = id_mat - rw_lap_mat
        lambda_max_rw = eigsh(rw_lap_mat, k=1, which='LM', return_eigenvectors=False)[0]
        wid_rw_normd_lap_mat = 2 * rw_normd_lap_mat / lambda_max_rw - id_mat
        return wid_rw_normd_lap_mat
    elif mat_type == 'hat_rw_normd_lap_mat':
        # For GCNConv
        wid_deg_mat = deg_mat + id_mat
        wid_adj_mat = adj_mat + id_mat
        hat_rw_normd_lap_mat = np.matmul(np.linalg.matrix_power(wid_deg_mat, -1), wid_adj_mat)
        return hat_rw_normd_lap_mat
    else:
        raise ValueError(f'ERROR: {mat_type} is unknown.')


# if __name__ == '__main__':
#     import json
#     dst_dir = './data/Porto/train/'
#     json_dir = './data/Porto/extro_data/'
#     with open(json_dir + 'new2raw_rid.json', "r") as f:
#         new2raw_rid = json.load(f)
#     with open(json_dir + 'raw_rn_dict.json', "r") as f:
#         raw_rn_dict = json.load(f)
#     # 'min_lat':41.142,
#     # 'min_lng':-8.652,
#     # 'max_lat':41.174,
#     # 'max_lng':-8.578,
#     mbr = MBR(41.142, -8.652, 41.174, -8.578)
#     grid_size = 50
#     # Build POI checkin trajectory graph
#     # train_df = pd.read_csv(os.path.join(dst_dir, 'NYC_train.csv'))
#     print('Build global POI checkin graph -----------------------------------')
#     G = build_global_POI_checkin_graph(dst_dir, new2raw_rid, raw_rn_dict, mbr, grid_size)
#     nodelist = G.nodes()
#     print(len(nodelist))
#     A = nx.adjacency_matrix(G, nodelist=nodelist)
#     print(A.shape)
#     # Save graph to disk
#     save_G_path = './Porto/extro_data/'
#     print("save to pickle...")
#     save_graph_to_pickle(G, dst_dir=save_G_path)
#     print("save to csv...")
#     save_graph_to_csv(G, dst_dir=save_G_path)
#     print("save to edgelist...")
#     save_graph_edgelist(G, dst_dir=save_G_path)
#     print("ok")