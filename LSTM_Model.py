import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib import rnn

class Model(object):
    def __init__(self, sequence_length, cell_size, vectors):
        self.stock_x = tf.placeholder(tf.float32, shape=[None, sequence_length, 1], name='stock_x')
        self.stock_y = tf.placeholder(tf.int32, shape=[None, 2], name='stock_y')
        self.text_x = tf.placeholder(tf.int32, shape=[None, sequence_length, 220], name='text_x')

        # event embedding
        with tf.name_scope("embedding_vertices"):
            embedding_W = tf.get_variable("embedding_matrix", initializer=tf.constant(vectors, dtype=tf.float32),
                                          trainable=False)
            self.embedding_texts = tf.nn.embedding_lookup(embedding_W, self.text_x, name='embedded_vertices')

        # attention on texts
        self.pre = tf.reshape(self.embedding_texts, shape=[-1, 220, vectors.shape[-1]])
        self.average = tf.reduce_sum(self.pre,axis=1)
        # with tf.name_scope('Con1V'):
        #     filiter = tf.get_variable('kernel', initializer=tf.truncated_normal([3,128,128]))
        #     cnn_bias = tf.get_variable('cnn_bias', initializer=tf.constant(0.1,shape=[128]))
        #     h_conv1 = tf.nn.tanh(tf.nn.conv1d(self.pre, filiter, 1, 'SAME') + cnn_bias)
        #     max_pool = tf.reduce_max(h_conv1,axis=1)

        # self.embedding_texts_att = self.Attention_Layer(self.preAttention, "attention_part")
        self.embedding_texts_att = tf.reshape(self.average, shape=[-1, sequence_length, 128])

        # combine texts and stock: [batch_size, sequence_length, embedding+stock]
        self.combined_x = tf.concat([self.stock_x, self.embedding_texts_att], axis=-1)

        with tf.name_scope("LSTM"):
            lstm_cell = rnn.BasicLSTMCell(cell_size, name='lstm_cell')
            lstm_cell = rnn.DropoutWrapper(lstm_cell, output_keep_prob=0.8)
            output, states = tf.nn.dynamic_rnn(lstm_cell, self.combined_x, dtype=tf.float32)

        with tf.name_scope('output'):
            output_w = tf.get_variable("output_weight", shape=[cell_size, 2],
                                       initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
            output_b = tf.get_variable("output_bias", initializer=tf.constant([0.01] * 2))

            self.scores = tf.nn.xw_plus_b(states[-1], output_w, output_b, name="ouput_layer")
            self.output = tf.argmax(self.scores, axis=1)

        with tf.name_scope("loss_accuracy"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.stock_y)
            self.loss = tf.reduce_mean(losses)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.output, tf.argmax(self.stock_y, axis=1)), 'float'))

    def Attention_Layer(self, input_, name):
        """
        self-attention
        :param input_: output from last Bi-RNN
        :param name: For 'word' encoder or 'sentence' encoder
        :return: vector encoded
        """
        shape = input_.shape
        with tf.name_scope('%s_Attention' % name):
            weight = tf.get_variable("AttentionWeight_%s" % name,
                                     initializer=tf.truncated_normal([shape[-1].value], mean=0, stddev=0.01),
                                     dtype=tf.float32)
            # :[*batch_size, length_*, hidden_units * 2]
            h = fully_connected(input_, shape[-1].value, tf.nn.tanh)

            # :[*batch_size, length_*, 1]
            alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(weight, h), keepdims=True, axis=-1), axis=1,name='alpha')
            # alpha = fully_connected(h,1,tf.nn.softmax)

            # :[*batch_size, hidden_units*2]
            return tf.reduce_sum(tf.multiply(input_, alpha), axis=1)



