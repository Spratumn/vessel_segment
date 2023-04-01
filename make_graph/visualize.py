import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


STR_ELEM = np.array([[1,1,1],[1,1,0],[0,0,0]], dtype=np.bool) # eight-neighbors
#STR_ELEM = np.array([[0,1,0],[1,1,0],[0,0,0]], dtype=np.bool) # four-neighbors

# graph visualization
VIS_FIG_SIZE = (10,10)
VIS_NODE_SIZE = 50
VIS_ALPHA = 0.5 # (both for nodes and edges)
VIS_NODE_COLOR = ['b','r','y','g'] # tp/fp/fn(+tn)/tn
VIS_EDGE_COLOR = ['b','g','r'] # tp/fn/fp



def visualize_graph(im, graph, show_graph=False, save_graph=True, \
                    num_nodes_each_type=None, custom_node_color=None, \
                    tp_edges=None, fn_edges=None, fp_edges=None, \
                    save_path='graph.png'):

    plt.figure(figsize=VIS_FIG_SIZE)
    if im.dtype==np.bool:
        bg = im.astype(int)*255
    else:
        bg = im

    if len(bg.shape)==2:
        plt.imshow(bg, cmap='gray', vmin=0, vmax=255)
    elif len(bg.shape)==3:
        plt.imshow(bg)

    plt.axis('off')
    pos = {}
    node_list = list(graph.nodes)
    for i in node_list:
        pos[i] = [graph.nodes[i]['x'],graph.nodes[i]['y']]

    if custom_node_color is not None:
        node_color = custom_node_color
    else:
        if num_nodes_each_type is None:
            node_color = 'b'
        else:
            if not (graph.number_of_nodes()==np.sum(num_nodes_each_type)):
                raise ValueError('Wrong number of nodes')
            node_color = [VIS_NODE_COLOR[0]]*num_nodes_each_type[0] + [VIS_NODE_COLOR[1]]*num_nodes_each_type[1]

    nx.draw(graph, pos, node_color='green', edge_color='blue', width=1, node_size=10, alpha=VIS_ALPHA)
    #nx.draw(graph, pos, node_color='darkgreen', edge_color='black', width=3, node_size=30, alpha=VIS_ALPHA)
    #nx.draw(graph, pos, node_color=node_color, node_size=VIS_NODE_SIZE, alpha=VIS_ALPHA)

    if tp_edges is not None:
        nx.draw_networkx_edges(graph, pos,
                               edgelist=tp_edges,
                               width=3, alpha=VIS_ALPHA, edge_color=VIS_EDGE_COLOR[0])
    if fn_edges is not None:
        nx.draw_networkx_edges(graph, pos,
                               edgelist=fn_edges,
                               width=3, alpha=VIS_ALPHA, edge_color=VIS_EDGE_COLOR[1])
    if fp_edges is not None:
        nx.draw_networkx_edges(graph, pos,
                               edgelist=fp_edges,
                               width=3, alpha=VIS_ALPHA, edge_color=VIS_EDGE_COLOR[2])

    if save_graph:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    if show_graph:
        plt.show()

    plt.cla()
    plt.clf()
    plt.close()