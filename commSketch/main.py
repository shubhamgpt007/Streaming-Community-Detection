from commSketch import *
from linearStreaming import *
import networkx as nx
import time
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    fname = './communities.txt'
    res = {}
    cs = CommSketch()
    gph = nx.read_edgelist("./data/amazon.txt", create_using=nx.Graph(), nodetype=int, edgetype=int)
    edge_lst = list(gph.edges())
    start = time.perf_counter()
    for i in range(len(edge_lst)):
        on_receive(cs, edge_lst[i][0], edge_lst[i][1])

    run = time.perf_counter() - start
    print(f'Computational runtime of this algo: {round(run, 2)} seconds\n')
    for k in cs.forest.keys():
        cs.forest[k] = cs.community(k)

    for j, v in cs.forest.items():
        res[v] = [j] if v not in res.keys() else res[v] + [j]

    with open(fname, 'w') as file:
        for nested_list in res.values():
            for word in nested_list:
                file.write(str(word) + ' ')
            file.write('\n')