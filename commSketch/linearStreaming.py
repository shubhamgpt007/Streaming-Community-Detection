from commSketch import *
from scipy.special import comb


def inner_dens(n, m):
    if m == 0:
        return 0
    else:
        return m/comb(n, 2)


def outer_dens(n1, n2, m):
    return m / (n1 * n2)


def on_receive(cs, node1, node2):
    cs.update_sketch(node1, node2)
    key = cs.community(node1)
    sec_key = cs.community(node2)
    if ((inner_dens(cs.sparseMat[key][key][0], cs.sparseMat[key][key][1]) +
         inner_dens(cs.sparseMat[sec_key][sec_key][0], cs.sparseMat[sec_key][sec_key][1])) * .5 <=
            outer_dens(cs.sparseMat[key][key][0], cs.sparseMat[sec_key][sec_key][0],
                       cs.sparseMat[key][sec_key][1])):
        cs.merge_com(key, sec_key)


def on_query(cs):
    return cs.forest
