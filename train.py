from Evolution_data import data_tool
from LSTM_Model import Model

import tensorflow as tf
import datetime, time
import os
import numpy as np

class Training(data_tool, Model):
    def __init__(self):
        data_tool.__init__(self)
        with tf.Graph().as_default():
            self.sess = tf.Session()
            with self.sess.as_default():
                Model.__init__(self, sequence_length=5, cell_size=128, vectors=self.embedding_matrix)
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                lr = tf.train.exponential_decay(0.001, global_step=self.global_step, decay_steps=10000, decay_rate=1) #学习率衰减
                optimizer = tf.train.AdamOptimizer(lr)
                grads_and_vars = optimizer.compute_gradients(self.loss)
                self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

                # Summary for loss and accuracy
                loss_summary = tf.summary.scalar("loss", self.loss)
                acc_summary = tf.summary.scalar("accuracy", self.accuracy)

                # Train Summaries
                self.train_summary_op = tf.summary.merge([loss_summary, acc_summary])
                train_summary_dir = os.path.join( "summaries", "train")
                self.train_summary_writer = tf.summary.FileWriter(train_summary_dir, self.sess.graph)

                # Test Summaries
                self.test_summary_op = tf.summary.merge([loss_summary, acc_summary])
                test_summary_dir = os.path.join( 'summaries', 'test')
                self.test_summary_writer = tf.summary.FileWriter(test_summary_dir, self.sess.graph)

                # initialize
                self.sess.run(tf.global_variables_initializer())

                # generate batches
                batches_train = self.batches_generate(self.stock_x_train, self.stock_y_train, self.text_x_train,
                                                      epoch_size=15,
                                                      batch_size=10, shuffle=True)

                total = (len(self.stock_y_train) // 10 + 1) * 15
                # training on batches
                print("Total step:", total)
                for i, batch in enumerate(batches_train):
                    batch_x, batch_y, batch_category = batch
                    self.train_(batch_x, batch_y, batch_category, total)
                    current_step = tf.train.global_step(self.sess, self.global_step)
                    if i % 2 == 0 and i > 0:
                        print('\nEvaluation:\n')
                        self.test_()
                        # print("Writing model...\n")
                        # saver.save(self.sess, checkpoint_model_dir, global_step=current_step)
                    if current_step == 1800:
                        break

    def real_words_length(self, batches):
        return np.ceil([np.argmin(batch.tolist() + [0]) for batch in batches.reshape((-1, batches.shape[-1]))])

    def Evaluation_test(self, sess, window=500, save=None):
        # start testing and saving data
        data_size = len(self.test_x)
        result = []
        for i in range(data_size // window + 1):
            left_, right_ = i * window, min((i+1) * window, data_size)
            result.append(sess.run(self.output,
                                   feed_dict={self.input_x: self.test_x[left_:right_],
                                              self.keep_prob: 1.0,
                                              self.input_category: self.category_test[left_: right_]
                                              #self.real_length: self.real_words_length(self.test_x[left_:right_])
                                              }))
        result = np.concatenate(result, axis=0)
        print("Test data accuracy:", np.mean(np.equal(np.argmax(self.test_y, axis=1), result)))
        self.test['pred'] = result+1
        self.test.to_csv(os.path.join(self.outdir, 'reviewBiLSTM_runs', save, "reviewbiLSTM.tsv"), sep='\t')

    # define operations
    def train_(self, batch_x, batch_y, category, total):
        feed_dict = {self.stock_x: batch_x,
                     self.stock_y: batch_y,
                     self.text_x: category}

        loss, _, accuracy, step, summaries = self.sess.run(
            [self.loss, self.train_op, self.accuracy, self.global_step, self.train_summary_op],
            feed_dict=feed_dict)

        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}/{}, loss {:g}, acc {:g}".format(time_str, step, total, loss, accuracy))
        self.train_summary_writer.add_summary(summaries, step)

    def test_(self):
        feed_dict = {self.stock_x: self.stock_x_test,
                     self.stock_y: self.stock_y_test,
                     self.text_x: self.text_x_test}
        loss, accuracy, step, summaries = self.sess.run(
            [self.loss, self.accuracy, self.global_step, self.test_summary_op],
            feed_dict=feed_dict)

        time_str = datetime.datetime.now().isoformat()
        print("Test: {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        self.test_summary_writer.add_summary(summaries, step)



if __name__ == '__main__':
    Training()