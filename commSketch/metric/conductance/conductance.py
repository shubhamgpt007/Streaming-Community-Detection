from collections import namedtuple
import networkx as nx
from cdlib.utils import convert_graph_formats
import numpy as np
FitnessResult = namedtuple("FitnessResult", "min max score std")
FitnessResult.__new__.__defaults__ = (None,) * len(FitnessResult._fields)


def conductance(graph: nx.Graph, community, summary: bool = True) -> object:

    graph = convert_graph_formats(graph, nx.Graph)
    values = []
    for com in community:
        coms = nx.subgraph(graph, com)

        ms = len(coms.edges())
        #print(ms)
        edges_outside = 0
        for n in coms.nodes():
            neighbors = graph.neighbors(n)
            for n1 in neighbors:
                if n1 not in coms:
                    edges_outside += 1
        try:
            ratio = float(edges_outside) / ((2 * ms) + edges_outside)
        except:
            ratio = 0
        values.append(ratio)

    if summary:
        return FitnessResult(
            min=min(values), max=max(values), score=np.mean(values), std=np.std(values)
        )
    return values
