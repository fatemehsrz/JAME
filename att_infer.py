
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics, model_selection, pipeline

class LogRegressor(object):

     def logregression(embeddings, y_att):

        auc_collect=[]
        for att_num in range(y_att.shape[1]):
            y_feat=y_att[:, att_num]

            X = []
            y = []

            for u  in range(len( embeddings)):
                vec = [float(i) for i in embeddings[u]]
                X.append(vec)
                y.append(y_feat[u])
                #print( u,y_att[u])

            seed=0
            train_precent=0.8
            training_size = int(train_precent * len(X))
            #np.random.seed(seed)
            shuffle_indices = np.random.permutation(np.arange(len(X)))
            X_train = [X[shuffle_indices[i]] for i in range(training_size)]  #X[:training_size]
            y_train = [y[shuffle_indices[i]] for i in range(training_size)]  #y[:training_size]
            X_test = [X[shuffle_indices[i]] for i in range(training_size, len(X))]  # X[training_size:]
            y_test = [y[shuffle_indices[i]] for i in range(training_size, len(X))]  #y[training_size:]

            # Step 3: Create a model and fit it

            scaler = StandardScaler()
            lin_clf = LogisticRegression(C=1)
            clf = pipeline.make_pipeline(scaler, lin_clf)

            # Train classifier
            clf.fit(X_train, y_train)

            # Test classifier
            auc_test = metrics.scorer.roc_auc_scorer(clf, X_test, y_test)
            auc_collect.append(auc_test)


        auc_all = np.mean(auc_collect)


        return  auc_all


