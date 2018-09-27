import time
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score


class Model(object):
    
    def __init__(self, model_def, window_size=None):
        print('Initizaling Neural Network model...'),
        # define tensorflow graph
        self.is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
        
        if window_size != None:
            self.inputs = tf.placeholder(tf.float32, [None, window_size, 32], name='inputs')
            self.logits = model_def(self.inputs, window_size, self.is_training)
        else:
            self.inputs = tf.placeholder(tf.float32, [None, 32], name='inputs')
            self.logits = model_def(self.inputs)
        self.window_size = window_size
        
        self.labels = tf.placeholder(tf.float32, [None,  6], name='labels')
        self.preds = tf.nn.sigmoid(self.logits, name='predictions')      
        self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.labels))
        print('Model initizalized.')
    
    
    def set_data(self, train, valid):
        self.train = train
        self.valid = valid
        
    def fit(self, epochs, batch_size, batches_gen, lr=0.001):
        print('Starting learning...')
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.cost)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
        start_time = time.time()
        
        # loop through epochs
        for e in range(epochs):
            loss = 0
            n_total_batches = 0
            partial_loss = 0; partial_n_batches_count = 0
            print_partial_loss_every = 1000000//batch_size
#             print_validation_every = 5500000//batch_size # 5.5M is around 1/3 of an epoch
            # loop through training series (and subjects)
            for xx,yy in batches_gen(self.train, batch_size, self.window_size, shuffle=True):
                _, res = self.sess.run([self.optimizer, self.cost], feed_dict={self.inputs: xx, self.labels: yy, self.is_training: True}) # train (update weights)
                loss += res # calculate loss
                n_total_batches += 1
                partial_loss += res; partial_n_batches_count += 1 # calculate partial loss
                
                if n_total_batches % print_partial_loss_every == 0:
                    loss_print = partial_loss/partial_n_batches_count # get mean loss to make loss comparable among different batch sizes
                    print("Epoch: {},\tBatch: {},\tMean Loss (last batches): {}".format(e, n_total_batches, loss_print))
                    partial_loss = 0; partial_n_batches_count = 0
#                 if n_total_batches % print_validation_every == 0:
#                     print("Validating...")
#                     valid_mean_auc = self.validate(batch_size, batches_gen)
#                     print("Valid Mean AUC: {}".format(valid_mean_auc))
            
            # print train and valid information
            loss /= n_total_batches # get mean loss to make loss comparable among different batch sizes
            valid_mean_auc = self.validate(batch_size, batches_gen)
            print("-----")
            print("Epoch: {},\tBatch: ----,\tMean Loss (epoch): {},\tValid Mean AUC: {}".format(e, loss, valid_mean_auc))
            print("-----")

        train_time = time.time() - start_time
        print("Model was trained in {} seconds.".format(train_time))
        
        
    def validate(self, batch_size, batches_gen):
        """Calculates the mean ROC AUC."""
        total_predictions = []
        total_labels = pd.DataFrame()
        for xx,yy in batches_gen(self.valid, batch_size, self.window_size, shuffle=False):
            predictions = self.sess.run(self.preds, feed_dict={self.inputs: xx, self.is_training: False})
            total_predictions.extend(predictions)
            total_labels = pd.concat((total_labels, yy))
        mean_auc = roc_auc_score(total_labels, total_predictions, average='macro')
        return mean_auc