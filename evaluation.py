
from link_pred import Link_Prediction
from classify import read_node_label, Classifier
from att_infer import LogRegressor
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np


def node_classification( embeddings, label_path, name, size):

        X, Y = read_node_label( embeddings, label_path,)

        f_c=open('results/%s_classification_%d.txt'%(name, size), 'w')

        all_ratio=[]

        for tr_frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:

               print(" Training classifier using {:.2f}% nodes...".format(tr_frac * 100))
               clf = Classifier(embeddings=embeddings, clf=LogisticRegression(), name=name)
               results= clf.split_train_evaluate(X, Y, tr_frac)

               avg='macro'
               f_c.write(name+' train percentage: '+ str(tr_frac)+ ' F1-'+avg+ ' '+ str('%0.5f'%results[avg]))
               all_ratio.append(results[avg])
               f_c.write('\n')



def link_pred( edge_file, embeddings, name, size):

            clf = Link_Prediction(embeddings=embeddings, edge_file= edge_file)
            auc=clf.predict()

            functions = ["hadamard","average","l1","l2"]

            f_l = open('results/%s_linkpred_%d.txt' % ( name, size), 'w')

            for i in functions:
                print( name, i, '%.3f' % np.mean(auc[i]))

                f_l.write(name + ' ' + str(size) + ' ' + str(i) + ' link-pred AUC: ' + str('%.3f' % np.mean(auc[i])) )
                f_l.write('\n')



def attribute_infer(embeddings, feat_path, name, y_att, size):

       f_r = open('results/%s_infer_%d.txt' % (name, size), 'w')

       auc= LogRegressor.logregression( embeddings=embeddings, y_att=y_att)

       f_r.write(name +' emb size: '+ str(size) +' Infer AUC: ' + str('%0.5f' % auc))
       f_r.write('\n')

       print(name, 'emb size: ', size, ' final att infer AUC: ', str('%0.5f' % auc))




def plot_embeddings( embeddings,label_path, name):
        X, Y = read_node_label( embeddings,label_path)

        emb_list = []
        for k in X:
            emb_list.append(embeddings[k])
        emb_list = np.array(emb_list)

        model = TSNE(n_components=2)
        node_pos = model.fit_transform(emb_list)

        color_idx = {}
        for i in range(len(X)):
            color_idx.setdefault(Y[i][0], [])
            color_idx[Y[i][0]].append(i)

        for c, idx in color_idx.items():
            plt.scatter(node_pos[idx, 0], node_pos[idx, 1],label=c)  # c=node_colors)

        plt.axis('off')
        plt.legend(loc= 'upper right', prop={'size': 15}, bbox_to_anchor=(1.15, 1), ncol=1)
        #plt.title('%s graph '%name)
        plt.savefig('%s_vis.pdf'%(name),  bbox_inches='tight',dpi=100)




