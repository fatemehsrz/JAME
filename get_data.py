import numpy as np
import networkx as nx
import keras
import scipy.sparse as sp
import random

class Data:

    def __init__(self, edge_file, name):

        self.G = nx.read_edgelist(edge_file)
        self.num_nodes = len(self.G.nodes())
        self.name=name

    def create_adj_from_edgelist(self, path):
        """ dataset_str: arxiv-grqc, blogcatalog """
        dataset_path = path
        with open(dataset_path, 'r') as f:
            header = next(f)
            edgelist = []
            for line in f:
                i, j = map(int, line.split())
                edgelist.append((i, j))
        g = nx.Graph(edgelist)

        adj= sp.lil_matrix(nx.adjacency_matrix(g))
        #adj= nx.adjacency_matrix(g)
        return adj


    def adj_norm(self, path):

        g = nx.read_edgelist(path)
        adj=nx.adjacency_matrix(g)
        adj = sp.coo_matrix(adj)
        adj_ = adj + sp.eye(adj.shape[0])
        rowsum = np.array(adj_.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(
            degree_mat_inv_sqrt).tocoo()
        return adj_normalized


    def get_label(self, label_file):

        y=[]
        f_class= open(label_file, 'r')
        class_label={}

        for line in f_class:
           a=line.strip('\n').split(' ')
           #print(a[0],a[1] )
           class_label[int(a[0])]=int(a[1])
        for i in range(self.num_nodes) :

            y.append(class_label[i])

        y_class = keras.utils.to_categorical(y)
        y_class=np.array(y_class)
        return y_class


    def get_ratio_label(self, ratio, label_file):

        y=[]
        nodes=[]
        f_class= open(label_file, 'r')
        class_label={}

        for line in f_class:
           a=line.strip('\n').split(' ')
           #print(a[0],a[1] )
           class_label[int(a[0])]=int(a[1])
           nodes.append(int(a[0]))

        removed_num= int((1-ratio)*self.num_nodes)


        for i in range( removed_num ):
            #id= random.randint(0, len(class_label))
            id= random.choice(nodes)

            print('observed ratio:', ratio , 'observed labels:', self.num_nodes-removed_num,  'of all nodes: ', self.num_nodes )
            class_label[id]= 0

        for i in range(self.num_nodes):
            y.append(class_label[i])

        y_class = keras.utils.to_categorical(y)
        y_class=np.array(y_class)
        return y_class


    def get_feat(self, feat_file):

        feat_dict={}

        if self.name=='pubmed':

            f_feat = open(feat_file, 'r')

            for line in f_feat:

                vec=[]
                a = line.strip('\n').split(' ')

                #print(a[1:])

                for i in a[1:]:
                    if i=="0.0":
                        vec.append(0)
                    else:
                        vec.append(1)

                #print(vec)
                feat_dict[int(a[0])] = [i for i in vec]

        else:

            f_feat= open(feat_file, 'r')

            for line in f_feat:

               a=line.strip('\n').split(' ')
               feat_dict[int(a[0])] = [int(i) for i in a[1: -1]]



        y_feat=[]
        for i in range(self.num_nodes) :
           y_feat.append(feat_dict[i])
        y_feat=np.array(y_feat)
        keep_colunmn=[]


        for att_num in range(y_feat.shape[1]):

            pos_count=0
            for j in y_feat[:, att_num]:
                if j==1:
                    pos_count+=1
                    #print(j,pos_count )
                else:continue

            if pos_count>100:

                keep_colunmn.append(att_num)

            else:
                continue

        new_feat= y_feat[:, keep_colunmn]

        print(att_num, len(keep_colunmn))


        return new_feat







