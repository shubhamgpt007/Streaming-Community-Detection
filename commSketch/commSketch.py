import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
from scipy.special import comb
import time
import pickle
import csv
import ast


class CommSketch:

    def __init__(self):
        self.forest = {}
        self.sparseMat = {}

    # Find the root of the set in which element `k` belongs
    def community(self, k):
        # if `k` is root
        while k != self.forest[k]:
            self.forest[k] = self.forest[self.forest[k]]
            k = self.forest[k]
        # recur for the parent until we find the root
        return k

    def update_sketch(self, node1, node2):
        # print(com_lst.values())
        add_flag = False
        # for key,value in com_lst.copy().items():
        if node1 in self.forest.keys() and node2 in self.forest.keys() and (
                self.community(node1) == self.community(node2)):
            key = self.community(node1)
            self.sparseMat[key][key][1] += 1
            add_flag = True

        elif node1 in self.forest.keys():
            key = self.community(node1)
            # print(key,sec_key)
            if node2 in self.forest.keys():
                sec_key = self.community(node2)
                add_flag = True
                # if key is not present in keys
                if key not in self.sparseMat[sec_key].keys():
                    self.sparseMat[sec_key][key] = [0, 1]
                else:
                    # print(key,sec_key)
                    self.sparseMat[sec_key][key][1] += 1
                    # print("0",dens_dic)

                if sec_key not in self.sparseMat[key].keys():
                    self.sparseMat[key][sec_key] = [0, 1]
                else:
                    self.sparseMat[key][sec_key][1] += 1
                # check for merge
                #merge_com(com_lst, dens_dic, key, sec_key, edge_lst[i][1])
            else:
                add_flag = True
                self.forest[node2] = node2
                #size[edge_lst[i][1]] = 1
                self.sparseMat[node2] = {node2: [1, 0]}
                self.sparseMat[node2].update({key: [0, 1]})
                self.sparseMat[key].update({node2: [0, 1]})
                # check for merge
                #merge_com(com_lst, dens_dic, key, edge_lst[i][1], None)

        elif node2 in self.forest.keys():
            key = self.community(node2)
            if node1 in self.forest.keys():
                sec_key = self.community(node1)
                add_flag = True
                # if key is not present in keys
                if key not in self.sparseMat[sec_key].keys():
                    self.sparseMat[sec_key][key] = [0, 1]
                else:
                    self.sparseMat[sec_key][key][1] += 1

                if sec_key not in self.sparseMat[key].keys():
                    self.sparseMat[key][sec_key] = [0, 1]
                else:
                    self.sparseMat[key][sec_key][1] += 1
                    # print("1",dens_dic)
                # check for merge
                #merge_com(com_lst, dens_dic, key, sec_key, edge_lst[i][0])
                # break
            else:
                add_flag = True
                self.forest[node1] = node1
                #size[edge_lst[i][0]] = 1
                self.sparseMat[node1] = {node1: [1, 0]}
                self.sparseMat[node1].update({key: [0, 1]})
                self.sparseMat[key].update({node1: [0, 1]})

                # check for merge
                #merge_com(com_lst, dens_dic, key, edge_lst[i][0], None)

        elif not add_flag:
            node = node1 if node1 < node2 else node2
            self.forest[node1] = node
            self.forest[node2] = node
            #size[edge_lst[i][0]] = 2
            self.sparseMat[node] = {node: [2, 1]}

    def merge_com(self, key, sec_key):
        if self.sparseMat[key][key][0] >= self.sparseMat[sec_key][sec_key][0]:
            self.sparseMat[key][key] = [(self.sparseMat[key][key][0] + self.sparseMat[sec_key][sec_key][0]),
                                        (self.sparseMat[key][key][1] + self.sparseMat[sec_key][sec_key][1]
                                         + self.sparseMat[key][sec_key][1])]
            self.forest[sec_key] = key
            # logic for replacing com_cnt with key
            if len(self.sparseMat[sec_key]) > 2:
                self.repl_key(key, sec_key)
        else:
            self.sparseMat[sec_key][sec_key] = [(self.sparseMat[sec_key][sec_key][0] + self.sparseMat[key][key][0]),
                                                (self.sparseMat[sec_key][sec_key][1] + self.sparseMat[key][key][1] +
                                                 self.sparseMat[sec_key][key][1])]
            self.forest[key] = sec_key
            if len(self.sparseMat[key]) > 2:
                self.repl_key(sec_key, key)

    def repl_key(self, key, sec_key):
        self.sparseMat[key].pop(sec_key)
        sec_key_data = self.sparseMat.pop(sec_key)
        for k, v in sec_key_data.items():
            if k == key or k == sec_key:
                continue
            if k in self.sparseMat[key].keys():
                self.sparseMat[key][k] = [(self.sparseMat[key][k][0] + v[0]),
                                          (self.sparseMat[key][k][1] + v[1])]
            else:
                self.sparseMat[key][k] = sec_key_data[k]

            self.sparseMat[k][key] = self.sparseMat[key][k]
            self.sparseMat[k].pop(sec_key)

