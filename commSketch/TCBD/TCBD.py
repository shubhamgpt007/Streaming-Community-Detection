import itertools
import math
import pickle
import networkx as nx
from networkx.utils import py_random_state
import networkx.algorithms.community as nx_comm
from cdlib.utils import convert_graph_formats
from collections import namedtuple
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
FitnessResult = namedtuple("FitnessResult", "min max score std")
FitnessResult.__new__.__defaults__ = (None,) * len(FitnessResult._fields)

"""Generators for classes of graphs used in studying social networks."""
__all__ = [
    "LFR_benchmark_graph",
    "TCBD_Graph"
]


def _zipf_rv_below(gamma, xmin, threshold, seed):
    result = nx.utils.zipf_rv(gamma, xmin, seed)
    while result > threshold:
        result = nx.utils.zipf_rv(gamma, xmin, seed)
    return result


def _powerlaw_sequence(gamma, low, high, condition, length, max_iters, seed):
    for i in range(max_iters):
        seq = []
        while not length(seq):
            # print(seed)
            seq.append(_zipf_rv_below(gamma, low, high, seed))
        if condition(seq):
            return seq
    raise nx.ExceededMaxIterations("Could not create power law sequence")


def _hurwitz_zeta(x, q, tolerance):
    z = 0
    z_prev = -float("inf")
    k = 0
    while abs(z - z_prev) > tolerance:
        z_prev = z
        z += 1 / ((k + q) ** x)
        k += 1
    return z


def _generate_min_degree(gamma, average_degree, max_degree, tolerance, max_iters):
    """Returns a minimum degree from the given average degree."""
    # Defines zeta function whether or not Scipy is available
    try:
        from scipy.special import zeta
    except ImportError:

        def zeta(x, q):
            return _hurwitz_zeta(x, q, tolerance)

    min_deg_top = max_degree
    min_deg_bot = 1
    min_deg_mid = (min_deg_top - min_deg_bot) / 2 + min_deg_bot
    itrs = 0
    mid_avg_deg = 0
    while abs(mid_avg_deg - average_degree) > tolerance:
        if itrs > max_iters:
            raise nx.ExceededMaxIterations("Could not match average_degree")
        mid_avg_deg = 0
        for x in range(int(min_deg_mid), max_degree + 1):
            mid_avg_deg += (x ** (-gamma + 1)) / zeta(gamma, min_deg_mid)
        if mid_avg_deg > average_degree:
            min_deg_top = min_deg_mid
            min_deg_mid = (min_deg_top - min_deg_bot) / 2 + min_deg_bot
        else:
            min_deg_bot = min_deg_mid
            min_deg_mid = (min_deg_top - min_deg_bot) / 2 + min_deg_bot
        itrs += 1
    # return int(min_deg_mid + 0.5)
    return round(min_deg_mid)


def _generate_communities(degree_seq, community_sizes, mu, max_iters, seed):
    # This assumes the nodes in the graph will be natural numbers.
    result = [set() for _ in community_sizes]
    n = len(degree_seq)
    free = list(range(n))
    for i in range(max_iters):
        v = free.pop()
        c = seed.choice(range(len(community_sizes)))
        # s = int(degree_seq[v] * (1 - mu) + 0.5)
        s = round(degree_seq[v] * (1 - mu))
        # If the community is large enough, add the node to the chosen
        # community. Otherwise, return it to the list of unaffiliated
        # nodes.
        if s < community_sizes[c]:
            result[c].add(v)
        else:
            free.append(v)
        # If the community is too big, remove a node from it.
        if len(result[c]) > community_sizes[c]:
            free.append(result[c].pop())
        if not free:
            return result
    msg = "Could not assign communities; try increasing min_community"
    raise nx.ExceededMaxIterations(msg)


@py_random_state(11)
def LFR_benchmark_graph(
        n,
        tau1,
        tau2,
        mu,
        average_degree=None,
        min_degree=None,
        max_degree=None,
        min_community=None,
        max_community=None,
        tol=1.0e-7,
        max_iters=500,
        seed=None,
):
    # Perform some basic parameter validation.
    if not tau1 > 1:
        raise nx.NetworkXError("tau1 must be greater than one")
    if not tau2 > 1:
        raise nx.NetworkXError("tau2 must be greater than one")
    if not 0 <= mu <= 1:
        raise nx.NetworkXError("mu must be in the interval [0, 1]")

    # Validate parameters for generating the degree sequence.
    if max_degree is None:
        max_degree = n
    elif not 0 < max_degree <= n:
        raise nx.NetworkXError("max_degree must be in the interval (0, n]")
    if not ((min_degree is None) ^ (average_degree is None)):
        raise nx.NetworkXError(
            "Must assign exactly one of min_degree and" " average_degree"
        )
    if min_degree is None:
        min_degree = _generate_min_degree(
            tau1, average_degree, max_degree, tol, max_iters
        )
    # print(seed)
    # Generate a degree sequence with a power law distribution.
    low, high = min_degree, max_degree

    def condition(seq):
        return sum(seq) % 2 == 0

    def length(seq):
        return len(seq) >= n

    deg_seq = _powerlaw_sequence(tau1, low, high, condition, length, max_iters, seed)
    # print(deg_seq)

    # Validate parameters for generating the community size sequence.
    if min_community is None:
        min_community = min(deg_seq)
    if max_community is None:
        max_community = max(deg_seq)

    # Generate a community size sequence with a power law distribution.
    #
    # TODO The original code incremented the number of iterations each
    # time a new Zipf random value was drawn from the distribution. This
    # differed from the way the number of iterations was incremented in
    # `_powerlaw_degree_sequence`, so this code was changed to match
    # that one. As a result, this code is allowed many more chances to
    # generate a valid community size sequence.
    low, high = min_community, max_community

    # print(low,high)

    def condition(seq):
        return sum(seq) == n

    def length(seq):
        return sum(seq) >= n

    # print(condition)
    comms = _powerlaw_sequence(tau2, low, high, condition, length, max_iters, seed)

    # print(comms)
    # Generate the communities based on the given degree sequence and
    # community sizes.
    max_iters *= 10 * n
    communities = _generate_communities(deg_seq, comms, mu, max_iters, seed)
    # print(communities)

    # Finally, generate the benchmark graph based on the given
    # communities, joining nodes according to the intra- and
    # inter-community degrees.
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for c in communities:
        for u in c:
            while G.degree(u) < round(deg_seq[u] * (1 - mu)):
                v = seed.choice(list(c))
                G.add_edge(u, v)
            while G.degree(u) < deg_seq[u]:
                v = seed.choice(range(n))
                if v not in c:
                    G.add_edge(u, v)
            G.nodes[u]["community"] = c
    return G


def _generate_temporal_communities(G, degree_seq, community_sizes, mu, max_iters, seed):
    # This assumes the nodes in the graph will be natural numbers.
    result = [set() for _ in community_sizes]
    old_fset_comms = {frozenset(G.nodes[v]["community"]) for v in G}
    old_comms = [set(x) for x in old_fset_comms]
    comms_ele = [s for s in range(len(old_comms)) for i in old_comms[s]]
    # homeless_ls = []
    trial_cnt = 0
    n = len(degree_seq)
    free = list(range(n))
    for i in range(max_iters):
        v = free.pop()
        if trial_cnt == 0:
            t = community_sizes.copy()
        if t:
            c = min(t)
            ind = [k for k, x in enumerate(community_sizes) if x == c]
            index = seed.choice(ind)

        s = round(degree_seq[v] * (1 - mu))
        # If the community is large enough, add the node to the chosen
        # community. Otherwise, return it to the list of unaffiliated
        # nodes.
        # if t:
        if s < c:
            # print(v,G.number_of_nodes()+v,s,c,result[index])
            result[index].add(G.number_of_nodes() + v)
            trial_cnt = 0
        else:
            free.append(v)
            trial_cnt += 1
            t.remove(c)
        # If the community is too big, remove a node from it.
        if len(result[index]) > c:
            free.append(result[index].pop() - G.number_of_nodes())
            trial_cnt += 1
            t.remove(c)

        if not (t):
            y = free.pop()
            rm_flag = False
            while not (rm_flag):
                oc = seed.choice(comms_ele)
                if round(degree_seq[y] * (1 - mu)) < len(old_comms[oc]):
                    # print(y,G.number_of_nodes()+y,s,len(old_comms[oc]),old_comms[oc])
                    old_comms[oc].add(G.number_of_nodes() + y)
                    # homeless_ls.append(G.number_of_nodes()+y)
                    rm_flag = True

            trial_cnt = 0
        if not free:
            return old_comms, result
    msg = "Could not assign communities; try increasing min_community"
    raise nx.ExceededMaxIterations(msg)


@py_random_state(12)
def TCBD_Graph(
        G,
        n,
        tau1,
        tau2,
        mu,
        average_degree=None,
        min_degree=None,
        max_degree=None,
        min_community=None,
        max_community=None,
        tol=1.0e-7,
        max_iters=500,
        seed=None,
):
    # Perform some basic parameter validation.
    if not tau1 > 1:
        raise nx.NetworkXError("tau1 must be greater than one")
    if not tau2 > 1:
        raise nx.NetworkXError("tau2 must be greater than one")
    if not 0 <= mu <= 1:
        raise nx.NetworkXError("mu must be in the interval [0, 1]")

    # Validate parameters for generating the degree sequence.
    if max_degree is None:
        max_degree = n
    elif not 0 < max_degree <= n:
        raise nx.NetworkXError("max_degree must be in the interval (0, n]")
    if not ((min_degree is None) ^ (average_degree is None)):
        raise nx.NetworkXError(
            "Must assign exactly one of min_degree and" " average_degree"
        )
    if min_degree is None:
        min_degree = _generate_min_degree(
            tau1, average_degree, max_degree, tol, max_iters
        )

    # Generate a degree sequence with a power law distribution.
    low, high = min_degree, max_degree

    def condition(seq):
        return sum(seq) % 2 == 0

    def length(seq):
        return len(seq) >= n

    deg_seq = _powerlaw_sequence(tau1, low, high, condition, length, max_iters, seed)
    # print(deg_seq)

    # Validate parameters for generating the community size sequence.
    if min_community is None:
        min_community = min(deg_seq)
    if max_community is None:
        max_community = max(deg_seq)

    # Generate a community size sequence with a power law distribution.
    #
    # TODO The original code incremented the number of iterations each
    # time a new Zipf random value was drawn from the distribution. This
    # differed from the way the number of iterations was incremented in
    # `_powerlaw_degree_sequence`, so this code was changed to match
    # that one. As a result, this code is allowed many more chances to
    # generate a valid community size sequence.
    low, high = min_community, max_community

    # print(low,high)

    def condition(seq):
        return (sum(seq) == n) ^ (sum(seq) < n)

    def length(seq):
        return len(seq) == math.floor(n / (0.5 * (min(n, high) + low)))

    # print(condition)
    comms = _powerlaw_sequence(tau2, low, high, condition, length, max_iters, seed)
    # Generate the communities based on the given degree sequence and
    # community sizes.
    max_iters *= 10 * n
    old_communities, new_communities = _generate_temporal_communities(G, deg_seq, comms, mu, max_iters, seed)
    # print(len(communities))

    # Finally, generate the benchmark graph based on the given
    # communities, joining nodes according to the intra- and
    # inter-community degrees.
    # G = nx.Graph()
    pre_gph_node_cnt = G.number_of_nodes()
    G.add_nodes_from(range(pre_gph_node_cnt, pre_gph_node_cnt + n))

    for old in old_communities:
        for u in old:
            if pre_gph_node_cnt <= u < pre_gph_node_cnt + n:
                ch = list(old)
                while G.degree(u) < round(deg_seq[u - pre_gph_node_cnt] * (1 - mu)):
                    # print(G.degree(u),round(deg_seq[u-pre_gph_node_cnt] * (1 - mu)))
                    v = seed.choice(ch)
                    G.add_edge(u, v)
                    if G.subgraph(old).degree(v) >= round(G.degree(v) * (1 - mu)):
                        while (round(G.degree(v) * (1 - mu)) + (G.degree(v) - G.subgraph(old).degree(v))) < G.degree(v):
                            x = seed.choice(range(pre_gph_node_cnt + n))
                            if x not in old:
                                G.add_edge(x, v)
                    ch.remove(v)
                while G.degree(u) < deg_seq[u - pre_gph_node_cnt]:
                    v = seed.choice(range(pre_gph_node_cnt + n))
                    if v not in old:
                        G.add_edge(u, v)
                G.nodes[u]["community"] = old
            else:
                G.nodes[u]["community"] = old

    for new in new_communities:
        for u in new:
            ch = list(new)
            while G.degree(u) < round(deg_seq[u - pre_gph_node_cnt] * (1 - mu)):
                v = seed.choice(ch)
                G.add_edge(u, v)
                ch.remove(v)
            while G.degree(u) < deg_seq[u - pre_gph_node_cnt]:
                v = seed.choice(range(pre_gph_node_cnt + n))
                if v not in new:
                    G.add_edge(u, v)
            G.nodes[u]["community"] = new

    return G

