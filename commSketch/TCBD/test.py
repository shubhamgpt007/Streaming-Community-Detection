from TCBD.TCBD import *


def generate_lfr_graph(mu, size=250):
    params = {"n": size, "tau1": 2, "tau2": 1.1, "mu": mu, "min_degree": 20, "max_degree": 50}

    G = LFR_benchmark_graph(params["n"], params["tau1"], params["tau2"], params["mu"],
                        min_degree=params["min_degree"],
                        max_degree=params["max_degree"],
                        max_iters=5000, seed=10)
    return G


def generate_TCBD(G, mu, size=250):
    params = {"n": size, "tau1": 2, "tau2": 1.1, "mu": mu, "min_degree": 20, "max_degree": 50}

    G = TCBD_Graph(G, params["n"], params["tau1"], params["tau2"], params["mu"],
                        min_degree=params["min_degree"],
                        max_degree=params["max_degree"],
                        max_iters=5000, seed=10)
    return G