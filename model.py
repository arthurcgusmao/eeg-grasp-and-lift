import os
import time
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score


class Model(object):
    
    def __init__(self, model_def, window_size=None):
        """Arguments:
        
        - `model_def`: can be either a model definition (see `model_defs.py` file) or a string corresponding to a path to an exported model.
        """
        if type(model_def) == str:
            print('Restoring model from disk, path=`{}`'.format(model_def))
            self.results_dir = os.path.abspath(model_def)
            self.tf_model_path = os.path.join(self.results_dir, 'tf-model')
            self.model_info_filepath = os.path.join(self.results_dir, 'model_info.tsv')
            self.lock_training = True
            
            self.model_info = pd.read_csv(self.model_info_filepath, sep='\t').loc[0].to_dict()
            self.window_size = self.model_info['window_size']
            
            tf.reset_default_graph()
            self.sess = tf.Session()
            new_saver = tf.train.import_meta_graph(self.tf_model_path + '.meta', clear_devices=True)
            new_saver.restore(self.sess, tf.train.latest_checkpoint(self.results_dir))
            
            graph = tf.get_default_graph()
            self.inputs = graph.get_tensor_by_name('inputs:0')
            self.preds = graph.get_tensor_by_name('predictions:0')
            self.logits = graph.get_tensor_by_name('out_logits_1:0')
            self.labels = graph.get_tensor_by_name('labels:0')
            self.is_training = graph.get_tensor_by_name('is_training:0')
        else:
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
            tf.identity(self.logits, name="out_logits_1")

            self.labels = tf.placeholder(tf.float32, [None,  6], name='labels')
            self.preds = tf.nn.sigmoid(self.logits, name='predictions')      
            self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.labels))
            print('Model initizalized.')

            self.results_dir = os.path.abspath('./results/' + str(int(time.time())))
            os.mkdir(self.results_dir)
            print('Model results directory created: {}'.format(self.results_dir))
            self.learning_curve_filepath = os.path.join(self.results_dir, 'learning_curve.tsv')
            self.model_info_filepath = os.path.join(self.results_dir, 'model_info.tsv')
            self.model_info = {
                'name': model_def.__name__,
                'window_size': window_size,
            }
            self.tf_model_path = os.path.join(self.results_dir, 'tf-model')
            self.lock_training = False
        
    def set_data(self, train, valid):
        self.train = train
        self.valid = valid
        self.model_info['# train'] = len(train)
        self.model_info['# valid'] = len(valid)
        
    def fit(self, epochs, batch_size, batches_gen, lr=0.001):
        if self.lock_training:
            print('Training locked!')
            return
        self.lock_training = True
        
        print('Starting learning...')
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.cost)
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
        start_time = time.time()
        learning_curve = []
        self.model_info['batch_size'] = batch_size
        self.model_info['lr'] = lr
        pd.DataFrame([self.model_info]).to_csv(self.model_info_filepath, sep='\t') # save model info in results dir
        
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
                    learning_curve.append({
                        'Epoch': e,
                        'Batch': n_total_batches,
                        'Mean Loss (last batches)': loss_print,
                    })
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
            learning_curve.append({
                'Epoch': e,
                'Mean Loss (epoch)': loss,
                'Valid Mean AUC': valid_mean_auc,
            })
            pd.DataFrame(learning_curve).to_csv(self.learning_curve_filepath, sep='\t') # save learning curve after each epoch

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
    
    
    def save_model(self):
        """Exports a model to disk"""
        self.saver.save(self.sess, self.tf_model_path)
#         tf.train.export_meta_graph(filename=self.tf_model_path)
        print('\nModel exported to {}'.format(self.results_dir))