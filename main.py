
import keras
from keras.layers import *
from keras.models import Model
import numpy as np
from get_data import Data
import evaluation as Eval
from keras import backend as K
import tensorflow as tf
from loss_weighting import LossWeighter
import time

import matplotlib.pylab as plt


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def reconstruction_loss(x):
    y_true, y_pred= x
    mask = K.not_equal(y_true, -1.0)
    mask = K.cast(mask, dtype=np.float32)
    num_examples = K.sum(mask, axis=1)
    pos = K.cast(K.equal(y_true, 1.0), dtype=np.float32)
    num_pos = K.sum(pos, axis=None)
    neg = K.cast(K.equal(y_true, 0.0), dtype=np.float32)
    num_neg = K.sum(neg, axis=None)
    pos_ratio = 1.0 - num_pos / num_neg
    mbce = mask * tf.nn.weighted_cross_entropy_with_logits(
        targets=y_true,
        logits=y_pred,
        pos_weight=pos_ratio
    )
    mbce = K.sum(mbce, axis=1) / num_examples
    return K.mean(mbce, axis=-1)


def class_loss(x):
    y_true, y_pred = x
    loss = K.sum( K.categorical_crossentropy(y_true, y_pred), axis=-1)
    return loss


def bin_loss(x):
    y_true, y_pred = x
    loss = K.sum( K.binary_crossentropy(y_true, y_pred), axis=-1)
    return loss

def ignoreLoss(true,pred):
    return pred #this just tries to minimize the prediction without any extra computation


###### Joint Autoencoders ######

def model(adj, y_class, y_feat, dim_size, epoch):

    time_callback = TimeHistory()

    trueClass_1 = Input((adj.shape[1],))
    trueClass_2 = Input((y_class.shape[1],))
    trueClass_3 = Input((y_feat.shape[1],))
    dummyOut = np.zeros((adj.shape[1],))

    X_a_input = Input(shape=(adj.shape[1],))
    X_b_input = Input(shape=(y_class.shape[1],))
    X_c_input = Input(shape=(y_feat.shape[1],))

    X_a_encoded = Dense(300, activation='relu')(X_a_input)

    X_b_encoded = Dense(300, activation='relu')(X_b_input)
    X_c_encoded = Dense(300, activation='relu')(X_c_input)

    shared_input = Add()([X_a_encoded , X_b_encoded, X_c_encoded])
    shared_output = Dense(dim_size, activation='sigmoid')(shared_input)


    X_a_decoded = Dense(300, activation='relu')(shared_output)  # ,kernel_regularizer=l2(0.01)
    X_a_decoded = Dense(adj.shape[1], activation='sigmoid' )( X_a_decoded)  #r_dot = Dot(axes=-1, normalize=True)


    X_b_decoded = Dense(300, activation='relu')(shared_output)
    X_b_decoded = Dense(y_class.shape[1], activation='sigmoid')(X_b_decoded )


    X_c_decoded = Dense(300, activation='relu')(shared_output)
    X_c_decoded = Dense(y_feat.shape[1] , activation='sigmoid')(X_c_decoded)


    linkLoss = Lambda(reconstruction_loss, name='loss_link')([trueClass_1, X_a_decoded])
    classLoss = Lambda(class_loss, name='loss_cls')([trueClass_2, X_b_decoded])
    featLoss = Lambda(bin_loss, name='loss_feat')([trueClass_3, X_c_decoded])

    weightedLoss = LossWeighter(name='weighted_loss',nb_outputs=3)([linkLoss, classLoss, featLoss])

    autoencoder = Model([X_a_input, X_b_input, X_c_input, trueClass_1, trueClass_2, trueClass_3], weightedLoss)

    autoencoder.compile(optimizer='adam', loss=ignoreLoss)
 
    history = autoencoder.fit([adj, y_class, y_feat, adj, y_class, y_feat], dummyOut, epochs=epoch, batch_size=40, callbacks=[time_callback], verbose=1)

    weights= [K.get_value(weight) for weight in autoencoder.layers[-1].loss_weights ]

    # for embeddings
    encoder = Model([X_a_input, X_b_input, X_c_input], shared_output)
    pred=encoder.predict([adj,y_class, y_feat ])


    embeddings={}
    for i in range(pred.shape[0]):
         embeddings[i]=pred[i]


    return embeddings, weights, time_callback, history


if __name__ == '__main__':

    for name in ['cora', 'citeseer', 'pubmed']: 

        edge_path='%s/%s-edgelist.txt'%(name,name)
        label_path='%s/%s-label.txt'%(name,name)
        feat_path= '%s/%s-feature.txt'%(name,name)

        data = Data(edge_path, name)

        adj= data.create_adj_from_edgelist(edge_path)
        y_class=data.get_label(label_path)
        y_feat = data.get_feat(feat_path)


        print(adj. shape, y_class.shape, y_feat.shape )


        vis= False

        t1= time.time()

        for epoch in [10]:

            embeddings, weights, time_callback, history= model(adj, y_class, y_feat, 128, epoch)

            f_opt= open('loss/%s_weights_%d.txt'%(name, epoch), 'w')
            print( 'w_a: ',weights[0], ' w_y: ', weights[1], ' w_x: ', weights[2] )


            f_opt.write( 'w_a: '+str(weights[0])+ ' w_y: '+str(weights[1])+ ' w_x: '+str(weights[2]))
            f_opt.write('\n')
            f_opt.close()

            f_loss= open('loss/%s_loss_%d.txt'%(name, epoch), 'w')

            for t in history.history['loss']:
                 f_loss.write( str(t))
                 f_loss.write('\n')
            f_loss.close()

            f_time = open('results/%s_time_%d.txt' % (name, epoch), 'w')

            for t in time_callback.times:
                f_time.write('Epoch time: ' + str(t))
                f_time.write('\n')
            f_time.close()


        Eval.node_classification(embeddings, label_path, name, epoch)
        Eval.link_pred(edge_path, embeddings, name, epoch)
        Eval.attribute_infer(embeddings, feat_path, name, y_feat, epoch)

        if vis== True:
             Eval.plot_embeddings(embeddings, label_path, name)







