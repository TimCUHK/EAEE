## Edge Augmentation via Eigenvalue Elevation v2.0 ##
## Tim Li @ CUHK-DSE ##
## tianyi.li@cuhk.edu.hk ##
## July 2022 ##

import networkx as nx
import numpy as np
import math
import copy
import time

## Graph Initiation ##

G = nx.karate_club_graph() # sample simple graph
graph_name = 'karate_club'

    # Use sample graph data in 'data' folder [three data types --> .gml/.csv/.txt]
    # G = nx.read_gml('data/dolphins.gml')
    # G = nx.read_edgelist(open('data/tvshow_edges.csv', 'rb') , delimiter=',', create_using=nx.Graph(), nodetype = int)
    # G = nx.read_edgelist('data/CA-HepTh.txt', create_using = nx.Graph(), nodetype = int)
            
N = len(G)
edge_ori_count = len(G.edges())

## Parameters: community detection ##

community_detection_flag = 0   # 0: community existent; 1: community detection needed
communities = []
detection_method = 0           # 0: Girvan-Newman; 1: greedy modularity; 2: label propagation; 3: Louvain; 4: fluid

## Parameters: algorithm ##

h = 2
w_h = N

## Parameters: system ##

plot_eigenvalue_flag = 0
save_info_flag = 1
save_edge_flag = 1
save_info_header = 'output/i_' + str(int(time.time())) 
save_edge_header = 'output/e_' + str(int(time.time()))
start_time = time.time()

