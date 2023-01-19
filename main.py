## Edge Augmentation via Eigenvalue Elevation v2.0 ##
## Tim Li @ CUHK-DSE ##
## tianyi.li@cuhk.edu.hk ##
## July 2022 ##

import matplotlib.pyplot as plt
import networkx as nx
import scipy as sp
import scipy.linalg  
import numpy as np
import math
import copy
import time
import networkx.algorithms.community as nx_comm
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community import girvan_newman
from networkx.algorithms.community import asyn_fluidc
from community import community_louvain
from collections import defaultdict

#plt.close('all')

################################################

from params import *

print('[Program START]')
print('Graph Initiation. N = ' + str(N) + ' --- E = ' + str(edge_ori_count))

## Community Detection ##

if community_detection_flag == 1:
    
    if detection_method == 0:
        communities = girvan_newman(G)
        communities = tuple(sorted(c) for c in next(communities))
    if detection_method == 1:
        communities = greedy_modularity_communities(G)
    if detection_method == 2:
        communities = list(nx_comm.asyn_lpa_communities(G))
    if detection_method == 3:
        communities = community_louvain.best_partition(G)
        res = defaultdict(list)
        for key, val in sorted(communities.items()):
            res[val].append(key)
            communities = []
        for i in range(len(res)):
            communities.append(res[i])
    if detection_method == 4:
        communities = list(asyn_fluidc(G,2))

M = len(communities)   # From spectrum --> list_eigen_G = nx.laplacian_spectrum(G); M = sum(list_eigen_G<1e-6)

## Inter-community Edge Assignment ##

G_edge_intra = []
G_edge_inter = []

for e in G.edges():

    intra_flag = 0
    for com in range(len(communities)):
        if (e[0] in communities[com]) and (e[1] in communities[com]):
            G_edge_intra.append(e)
            intra_flag = 1
            break

    if intra_flag == 0:
        G_edge_inter.append(e)

G.remove_edges_from(G_edge_inter)

################################################

## Edge Augmentation via Eigenvalue Elevation ##

L = nx.laplacian_matrix(G)
lamda,U = scipy.linalg.eigh(L.todense())
list_eigen = nx.laplacian_spectrum(G)   # Normalized Laplacian --> list_eigen = sp.linalg.eigvalsh(nx.normalized_laplacian_matrix(G).todense())

# Augmentation #

L_prime = U @ (np.diag(lamda) + np.diag([0]+[w_h]*h + [0]*(N-h-1))) @ U.T
lamda_prime,U_prime = scipy.linalg.eigh(L_prime)
delta_d = np.round(np.diag(L_prime) - np.diag(L.todense()))
delta_d = [max(i,0) for i in delta_d]

list_node_adjust = []
for i in sorted(enumerate(delta_d), key=lambda x:x[1],reverse=True):
    if i[1] == 0:
        break
    list_node_adjust.append([list(G.nodes)[i[0]],i[1]])

print('Largest realisable degree: ' + str(len(list_node_adjust)))
print('Largest additional degree: ' + str(int(list_node_adjust[0][1])))

edge_aug = []
edge_count = []
edge_count_intra = []
edge_count_inter = []

for i in range(len(list_node_adjust)):
    edge_max = list_node_adjust[i][1]
    edge_real = []
    for j in range(len(list_node_adjust)-i-1):
        if (list_node_adjust[i][1]>0 and list_node_adjust[i+j+1][1]>0):
            
            if (list_node_adjust[i][0],list_node_adjust[i+j+1][0]) in G.edges():
                continue
            if (list_node_adjust[i+j+1][0],list_node_adjust[i][0]) in G.edges():
                continue
            
            edge_real.append((list_node_adjust[i][0],list_node_adjust[i+j+1][0]))
            
            if nx.has_path(G, list_node_adjust[i][0], list_node_adjust[i+j+1][0]):
                edge_count_intra.append((list_node_adjust[i][0],list_node_adjust[i+j+1][0]))
            else:
                edge_count_inter.append((list_node_adjust[i][0],list_node_adjust[i+j+1][0]))
                
            list_node_adjust[i][1] = list_node_adjust[i][1] - 1
            list_node_adjust[i+j+1][1] = list_node_adjust[i+j+1][1] - 1
            
    edge_aug = edge_aug + edge_real
    edge_count.append(len(edge_real))

print('Number of realized augmented edges: ' + str(sum(edge_count)))   
print('Realization ratio: ' + str(sum(edge_count)/(h*w_h/2)))   
print('Intra vs. inter edges: ' + str(len(edge_count_intra)) + '/' + str(len(edge_count_inter)))   
print('Inter edge ratio: ' + str(len(edge_count_inter)/sum(edge_count))) 
print('Theta: ' + str(1 + sum(edge_count)/edge_ori_count))

GG = G.copy()
GG.add_edges_from(edge_aug)
list_eigen_GG = nx.laplacian_spectrum(GG)
MM = sum(list_eigen_GG<1e-6)
print('Number of components: ' + str(M)  + '-->' + str(MM))

# Inter-community edge recovery #

edge_inter_new = []
for k in edge_count_inter:
    if k[0] < k[1]:
        edge_inter_new.append((k[0],k[1]))
    else:
        edge_inter_new.append((k[1],k[0]))

edge_inter_ori = []
for k in G_edge_inter:
    if k[0] < k[1]:
        edge_inter_ori.append((k[0],k[1]))
    else:
        edge_inter_ori.append((k[1],k[0]))

edge_recover = list(set(edge_inter_ori) & set(edge_inter_new))

print('Recovery Ratio: ' + str(len(edge_recover)) + '/' + str(len(edge_inter_ori)))

print('[Program END]')
print('Elapsed time = ' + str(int(time.time()-start_time)) + ' seconds.')

# Plot #

if plot_eigenvalue_flag == 1:
    
    plt.figure(figsize=(8,8)) 
    
    list_eigen_G = nx.laplacian_spectrum(G)
    GG.add_edges_from(edge_aug)
    list_eigen_GG = nx.laplacian_spectrum(GG)

    plt.plot(list_eigen_G,'*')
    plt.plot(list_eigen_GG,'*')

    plt.show()
    
# Save #

if save_info_flag == 1:
    
    f = open(save_info_header + '_info.txt','w')
    f.write('******** Save Result ********\n')
    f.write('*****************************\n')
    f.write('-----> Graph Info\n')
    f.write('Graph name: ' + str(graph_name) + '\n')
    f.write('N = ' + str(N) + ' --- E = ' + str(edge_ori_count) + '\n')
    f.write('Community detection method: ' + str(detection_method) + '\n')
    f.write('Number of communities: ' + str(len(communities)) + '\n')
    f.write('-----> Parameters\n')
    f.write('h = ' + str(h) + ' --- w_h = ' + str(w_h) + '\n')
    f.write('-----> Algorithm outcome\n')
    f.write('Largest realisable degree: ' + str(len(list_node_adjust)) + '\n')
    f.write('Largest additional degree: ' + str(int(list_node_adjust[0][1])) + '\n')
    f.write('Number of realized augmented edges: ' + str(sum(edge_count)) + '\n')   
    f.write('Realization ratio: ' + str(sum(edge_count)/(h*w_h/2)) + '\n')   
    f.write('Intra vs. inter edges: ' + str(len(edge_count_intra)) + '/' + str(len(edge_count_inter)) + '\n')   
    f.write('Inter edge ratio: ' + str(len(edge_count_inter)/sum(edge_count)) + '\n') 
    f.write('Theta: ' + str(1 + sum(edge_count)/edge_ori_count) + '\n')
    f.write('Number of components: ' + str(M)  + '-->' + str(MM) + '\n')
    f.write('Recovery Ratio: ' + str(len(edge_recover)) + '/' + str(len(edge_inter_ori)) + '\n')
    f.write('-----> Elapsed time = ' + str(int(time.time()-start_time)) + ' seconds.')
    f.close()

if save_edge_flag == 1:
    
    df = pd.DataFrame(edge_aug)
    df.to_excel(save_edge_header + '_new_edge.xlsx', index=False)
    