
import networkx as nx
from sklearn import pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import  metrics
from sklearn.metrics import average_precision_score


class Graph():

    def __init__(self,
                 nx_G=None, is_directed=False,
                 prop_pos=0.5, prop_neg=0.5,
                 workers=1,
                 random_seed=None):
        self.G = nx_G
        self.is_directed = is_directed
        self.prop_pos = prop_pos
        self.prop_neg = prop_neg
        self.wvecs = None
        self.workers = workers
        self._rnd = np.random.RandomState(seed=random_seed)

    def read_graph(self, input):

        G = nx.read_edgelist(input, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G.adj[edge[0]][edge[1]]['weight'] = 1

        print("Read graph, nodes: %d, edges: %d" % (G.number_of_nodes(), G.number_of_edges()))

        G1= G.to_undirected()
        self.G = G1


    def generate_pos_neg_links(self):

        # Select n edges at random (positive samples)
        n_edges = self.G.number_of_edges()
        n_nodes = self.G.number_of_nodes()
        npos = int(self.prop_pos * n_edges)
        nneg = int(self.prop_neg * n_edges)
        

        n_neighbors = [len(list(self.G.neighbors(v))) for v in self.G.nodes()] ##
        n_non_edges = n_nodes - 1 - np.array(n_neighbors)
        

        non_edges = [e for e in nx.non_edges(self.G)]
        print("Finding %d of %d non-edges" % (nneg, len(non_edges)))

        # Select m pairs of non-edges (negative samples)
        rnd_inx = self._rnd.choice(len(non_edges), nneg, replace=False)
        neg_edge_list = [non_edges[ii] for ii in rnd_inx]

        if len(neg_edge_list) < nneg:
            raise RuntimeWarning(
                "Only %d negative edges found" % (len(neg_edge_list))
            )

        print("Finding %d positive edges of %d total edges" % (npos, n_edges))

        # Find positive edges, and remove them.
        edges = self.G.edges()
        
        edges=list(edges)
        
        pos_edge_list = []
        n_count = 0
        n_ignored_count = 0
        
        
        rnd_inx = self._rnd.permutation(n_edges)
        
        for eii in rnd_inx.tolist():
            edge = edges[eii]

            # Remove edge from graph
            data = self.G[edge[0]][edge[1]]
            self.G.remove_edge(*edge)

            reachable_from_v1 = nx.connected._plain_bfs(self.G, edge[0])
            if edge[1] not in reachable_from_v1:
                self.G.add_edge(*edge, **data)
                n_ignored_count += 1
            else:
                pos_edge_list.append(edge)
                print("Found: %d edges   " % (n_count), end="\r")
                n_count += 1

            if n_count >= npos:
                break

        edges_num= len(pos_edge_list)
        self._pos_edge_list = pos_edge_list
        self._neg_edge_list = neg_edge_list
        
        print('pos_edge_list',len(self._pos_edge_list))
        print('neg_edge_list',len(self._neg_edge_list ))



    def get_selected_edges(self):
        edges = self._pos_edge_list + self._neg_edge_list
        labels = np.zeros(len(edges))
        labels[:len(self._pos_edge_list)] = 1
        return edges, labels


    def edges_to_features(self, edge_list, edge_function, emb_size, model):

        n_tot = len(edge_list)
        feature_vec = np.empty((n_tot, emb_size), dtype='f')
        
       
        for ii in range(n_tot):
            v1, v2 = edge_list[ii]
            
            # Edge-node features
            emb1 = np.asarray(model[v1])
            emb2 = np.asarray(model[v2])

            # Calculate edge feature
            feature_vec[ii] = edge_function(emb1, emb2)

        return feature_vec


class Link_Prediction(object):

    def __init__(self, embeddings, edge_file):
        self.embeddings = embeddings
        self.edge_file = edge_file

    def create_train_test_graphs(self, input= 'Facebook.edges',regen='regen',workers=8 ):
        
        
        default_params = {
        'edge_function': "hadamard",
        "prop_pos": 0.5,                # Proportion of edges to remove nad use as positive samples
        "prop_neg": 0.5,                # Number of non-edges to use as negative samples

        }

        # Remove half the edges, and the same number of "negative" edges
        prop_pos = default_params['prop_pos']
        prop_neg = default_params['prop_neg']

        print("Regenerating link prediction graphs")
        # Train graph embeddings on graph with random links
        Gtrain = Graph(is_directed=False,
                          prop_pos=prop_pos,
                          prop_neg=prop_neg,
                          workers=workers)
        Gtrain.read_graph(input)
        Gtrain.generate_pos_neg_links()

        # Generate a different random graph for testing
        Gtest = Graph(is_directed=False,
                         prop_pos=prop_pos,
                         prop_neg=prop_neg,
                         workers = workers)
        Gtest.read_graph(input)
        Gtest.generate_pos_neg_links()

        return Gtrain, Gtest

    
    
    def test_edge_functions(self ,num_experiments=2, emb_size=128, model=None,edges_train=None , edges_test=None, Gtrain=None, Gtest=None, labels_train=None, labels_test=None):
         
        edge_functions = {
            "hadamard": lambda a, b: a * b,
            "average": lambda a, b: 0.5 * (a + b),
            "l1": lambda a, b: np.abs(a - b),
            "l2": lambda a, b: np.abs(a - b) ** 2,
        }

        aucs = {func: [] for func in edge_functions}
        aps = {func: [] for func in edge_functions}


        for iter in range(num_experiments):
            print("Iteration %d of %d" % (iter, num_experiments))
    
            for edge_fn_name, edge_fn in edge_functions.items():
                
                #print(edge_fn_name, edge_fn)
                # Calculate edge embeddings using binary function
                edge_features_train = Gtrain.edges_to_features(edges_train, edge_fn, emb_size, model)
                edge_features_test = Gtest.edges_to_features(edges_test, edge_fn, emb_size, model)


                # Linear classifier
                scaler = StandardScaler()
                lin_clf = LogisticRegression(C=1.0)

                clf = pipeline.make_pipeline(scaler, lin_clf)

                # Train classifier
                clf.fit(edge_features_train, labels_train)
                AUC=  metrics.scorer.roc_auc_scorer(clf, edge_features_test, labels_test)
                aucs[edge_fn_name].append(AUC)

                #AP = average_precision_score(labels_test, clf.predict(edge_features_test))
                #aps[edge_fn_name].append(AP)



        return aucs
    
      
    def predict(self):
        
        Gtrain, Gtest = self.create_train_test_graphs(self.edge_file, regen='regen', workers=2)
    
         # Train and test graphs, with different edges
        edges_train, labels_train = Gtrain.get_selected_edges()
        edges_test, labels_test = Gtest.get_selected_edges()

        auc= self.test_edge_functions(2, 128, self.embeddings, edges_train , edges_test, Gtrain,  Gtest, labels_train ,labels_test)

        return auc





