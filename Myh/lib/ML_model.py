# coding:utf-8
import numpy as np
import utils as utils
import tensorflow as tf
import random
from math import exp
import sys
import math
import scipy
import scipy.io
import logging
from attention import attention
# from NormalizingRadialFlow import NormalizingRadialFlow
# from NormalizingPlanarFlow import NormalizingPlanarFlow

class Params:
    """Parameters for DMF
    """

    def __init__(self):
        self.a = 1
        self.b = 0.01
        self.lambda_u = 0.1
        self.lambda_v = 10
        self.lambda_r = 1
        self.max_iter = 10
        self.M = 300

        # params.lambda_u = 0.1
        # params.lambda_v = 10
        # params.lambda_r = 1
        # params.a = 1
        # params.b = 0.1
        # params.M = 300
        # params.n_epochs = 100
        # params.max_iter = 1

        # for updating W and b
        self.lr = 0.001
        self.batch_size = 32
        self.n_epochs = 100

        # params.lambda_u = 0.1
        # params.lambda_v = 10
        # params.lambda_r = 1
        # params.a = 1
        # params.b = 0.01
        # params.M = 300
        # params.n_epochs = 100
        # params.max_iter = 1

class CVAE_user:
    def __init__(self, nFlows, num_users, num_items, num_factors, params, input_dim,input_dim_user,
                 dims, activations, n_z=50, loss_type='cross-entropy', flow_type="Planar",
                 radial_flow_type='Given in the paper on NF', lr=0.1,
                 wd=1e-4, dropout=0.1, random_seed=0,print_step=50, verbose=True,
                 invert_condition=True, beta=True):

        # model_epoch_test = CVAE(num_users=5551, num_items=16980, num_factors=num_factors50, params=params,
        #              input_dim=8000, dims=[200, 100], n_z=num_factors, activations=['tanh', 'tanh'],
        #              loss_type='cross-entropy', lr=0.001, random_seed=0, print_step=10, verbose=False)
        self.nFlows = nFlows
        self.m_num_users = num_users
        self.m_num_items = num_items
        self.m_num_factors = num_factors  # 50
        self.input_dim_user = input_dim_user
        # BPR的placeholder####################################
        self.u = tf.placeholder(tf.int32, [None])
        self.i = tf.placeholder(tf.int32, [None])
        self.j = tf.placeholder(tf.int32, [None])
        # #### BPR############################################

        self.m_U = 0.1 * np.random.randn(self.m_num_users, self.m_num_factors)  # 5551x50 ##########################21:22
        self.m_theta_user = 0.1 * np.random.randn(self.m_num_users, self.m_num_factors)
        self.m_V = 0.1 * np.random.randn(self.m_num_items, self.m_num_factors)  # 16980x50
        self.m_theta = 0.1 * np.random.randn(self.m_num_items, self.m_num_factors)  # 16980x50
        self.item_bias = np.zeros(self.m_num_items)

        # BPR的user表示以及Item表示######################################
        self.u_emb = tf.nn.embedding_lookup(self.m_U, self.u)
        self.i_emb = tf.nn.embedding_lookup(self.m_V, self.i)
        self.j_emb = tf.nn.embedding_lookup(self.m_V, self.j)
        self.i_item_bias = tf.nn.embedding_lookup(self.item_bias, self.i)
        self.j_item_bias = tf.nn.embedding_lookup(self.item_bias, self.j)

        # MF predict: u_i > u_j
        self.x_BPR = tf.reduce_sum(tf.multiply(self.u_emb, (self.i_emb - self.j_emb)), 1, keep_dims=True)
        self.x_BPR_32 = tf.cast(self.x_BPR, dtype=tf.float32)
        # AUC for one user:reasonable iff all (u,i,j) pairs are from the same user
        # average AUC = mean( auc for each user in test set)
        self.mf_auc = tf.reduce_mean(tf.to_float(self.x_BPR_32 > 0))  # float32
        self.bias_regularization = 1.0
        # self.user_regularization = 0.0025
        # self.positive_item_regularization = 0.0025
        # self.negative_item_regularization = 0.00025
        self.user_regularization = 1  # 21:22
        self.positive_item_regularization = 0.1  # 21:22
        self.negative_item_regularization = 0.9  # 21:22
        l2_norm = tf.add_n([
            tf.reduce_sum(self.user_regularization * tf.multiply(self.u_emb, self.u_emb)),
            tf.reduce_sum(self.positive_item_regularization * tf.multiply(self.i_emb, self.i_emb)),
            tf.reduce_sum(self.negative_item_regularization * tf.multiply(self.j_emb, self.j_emb)),
            tf.reduce_sum(self.bias_regularization * self.i_item_bias**2),
            tf.reduce_sum(self.bias_regularization * self.j_item_bias ** 2)
        ])

        l2_norm_32 = tf.cast(l2_norm, dtype=tf.float32)
        self.bprloss = l2_norm_32 - tf.reduce_mean(tf.log(tf.sigmoid(self.x_BPR_32)))
        # train_op = tf.train.GradientDescentOptimizer(0.01).minimize(bprloss)
        # ##### BPR #############################################

        self.input_dim = input_dim  # 8000
        self.dims = dims  # [200, 100]
        self.activations = activations
        self.lr = lr  # 0.001
        self.params = params
        self.print_step = print_step  # 10
        self.verbose = verbose
        self.invert_condition = invert_condition
        self.radial_flow_type = radial_flow_type
        self.loss_type = loss_type
        self.flow_type = flow_type
        self.n_z = n_z  # 50
        self.weights = []
        self.reg_loss = 0

        self.x_user = tf.placeholder(tf.float32, [None, self.input_dim_user], name='x')
        self.x = tf.placeholder(tf.float32, [None, self.input_dim], name='x')  #
        self.v = tf.placeholder(tf.float32, [None, self.m_num_factors])  #
        self.v_user = tf.placeholder(tf.float32, [None, self.m_num_factors])

        self.z0, self.z_k, sum_log_detj = self.inference_generation(self.x, self.input_dim)
        x_recon, x_recons_mean, x_recons_logvar = self.generation(self.z_k)
        global_step = tf.Variable(0, trainable=False)
        self.recons_loss1, self.kl_loss1, self.loss_op = self.elbo_loss(self.x, x_recon, beta, global_step,
                            z_mu=self.z_mean, z_var=self.z_var, z0=self.z0,
                            zk=self.z_k, sum_log_detj=sum_log_detj)
        self.v_loss = 1.0 * params.lambda_v / params.lambda_r * tf.reduce_mean(
            tf.reduce_sum(tf.square(self.v - self.z0), 1))

        self.z_user, self.z_k_user, sum_log_detj_user = self.inference_generation_user(self.x_user)
        x_recon_user, x_recons_mean_user, x_recons_logvar_user = self.generation_user(self.z_k_user)
        global_step = tf.Variable(0, trainable=False)
        self.recons_loss_user, self.kl_loss1_user, self.loss_op_user\
            = self.elbo_loss(self.x_user, x_recon_user, beta, global_step,
                            z_mu=self.z_mean_user, z_var=self.z_var_user, z0=self.z_user,
                            zk=self.z_k_user, sum_log_detj=sum_log_detj_user)

        self.v_user_loss = 1.0 * params.lambda_v / params.lambda_r * tf.reduce_mean(
            tf.reduce_sum(tf.square(self.v_user - self.z_user), 1))

        self.loss = self.loss_op + self.v_loss + 2e-4 * self.reg_loss + self.loss_op_user + self.v_user_loss


        # train_op = tf.train.GradientDescentOptimizer(0.01).minimize(bprloss)
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        # Initializing the tensor flow variables
        self.saver = tf.train.Saver(self.weights)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        init = tf.global_variables_initializer()

        # Launch the session
        self.sess = tf.Session(config=config)
        self.sess.run(init)

    def inference_generation(self, x, input_dim):
        with tf.variable_scope("inference", reuse=tf.AUTO_REUSE):
            rec = {'W1': tf.get_variable("W1", [input_dim, self.dims[0]],  # W1 = 26x200
                                         initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                   'b1': tf.get_variable("b1", [self.dims[0]],  # b1 = 200
                                         initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                   'W2': tf.get_variable("W2", [self.dims[0], self.dims[1]],  # W2 = 200x100
                                         initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                   'b2': tf.get_variable("b2", [self.dims[1]],  # b2 = 100
                                         initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                   'W_z_mean': tf.get_variable("W_z_mean", [self.dims[1], self.n_z],  # 100x50
                                               initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                   'b_z_mean': tf.get_variable("b_z_mean", [self.n_z],  # 50
                                               initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                   # 'W_z_mean': tf.get_variable("W_z_mean", [self.dims[0], self.n_z],  # 100x50
                   #                             initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                   # 'b_z_mean': tf.get_variable("b_z_mean", [self.n_z],  # 50
                   #                             initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                   # 'W_z_log_sigma': tf.get_variable("W_z_log_sigma", [self.dims[0], self.n_z],  # 100x50
                   #                                  initializer=tf.contrib.layers.xavier_initializer(),
                   #                                  dtype=tf.float32),
                   # 'b_z_log_sigma': tf.get_variable("b_z_log_sigma", [self.n_z],  # 50
                   #                                  initializer=tf.constant_initializer(0.0), dtype=tf.float32)}
                   'W_z_log_sigma': tf.get_variable("W_z_log_sigma", [self.dims[1], self.n_z],  # 100x50
                                                    initializer=tf.contrib.layers.xavier_initializer(),
                                                    dtype=tf.float32),
                   'b_z_log_sigma': tf.get_variable("b_z_log_sigma", [self.n_z],  # 50
                                                    initializer=tf.constant_initializer(0.0), dtype=tf.float32)}
            with tf.variable_scope("normalizing_flow"):
                nf = {'w_us': tf.get_variable("w_us", [self.dims[1], self.nFlows*self.n_z],  # W1 = 8000x200
                                             initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                       'b_us': tf.get_variable("b_us", [self.nFlows*self.n_z],  # b1 = 200
                                             initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                       'w_ws': tf.get_variable("w_ws", [self.dims[1], self.nFlows*self.n_z],  # W2 = 200x100
                                             initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                       'b_ws': tf.get_variable("b_ws", [self.nFlows*self.n_z],  # b2 = 100
                                             initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                       'w_bs': tf.get_variable("w_bs", [self.dims[1], self.n_z],  # 100x50
                                                   initializer=tf.contrib.layers.xavier_initializer(),
                                                   dtype=tf.float32),
                       'b_bs': tf.get_variable("b_z_mean", [self.n_z],  # 50
                                                   initializer=tf.constant_initializer(0.0), dtype=tf.float32)}
        self.weights += [rec['W1'], rec['b1'], rec['W2'], rec['b2'], rec['W_z_mean'],
                         rec['b_z_mean'], rec['W_z_log_sigma'], rec['b_z_log_sigma']]
        # self.weights += [rec['W1'], rec['b1'], rec['W_z_mean'],
        #                  rec['b_z_mean'], rec['W_z_log_sigma'], rec['b_z_log_sigma']]
        self.reg_loss += tf.nn.l2_loss(rec['W1']) + tf.nn.l2_loss(rec['W2'])
        # self.reg_loss += tf.nn.l2_loss(rec['W1'])
        # self.x = tf.placeholder(tf.float32, [None, self.input_dim], name='x')
        h1 = tf.nn.sigmoid(
            tf.matmul(x, rec['W1']) + rec['b1'])  # 16980x200
        # attention_output, alphas = attention(h2, attention_size=50, keep_prob=0.5, return_alphas=True)
        h2 = self.activate(
            tf.matmul(h1, rec['W2']) + rec['b2'], self.activations[1])  # 16980x100
        # h2 = tf.reshape(h2, [-1, 1, self.dims[1]])
        # h1 = tf.reshape(h1, [-1, 1, 100])
        # attention_output, alphas = attention(h2, attention_size=50, keep_prob=0.5, time_major=True, return_alphas=True)
        # attention_output, alphas = attention(h1, attention_size=50, keep_prob=0.5, time_major=True, return_alphas=True)

        self.z_mean = tf.matmul(h2, rec['W_z_mean']) + rec['b_z_mean']  # 16980x50#############
        self.z_log_sigma_sq = tf.matmul(h2, rec['W_z_log_sigma']) + rec['b_z_log_sigma']  # 16980x50################
        self.z_var = tf.exp(self.z_log_sigma_sq)

        # Normalizing Flow parameters
        us = tf.matmul(h2, nf['w_us']) + nf['b_us']
        ws = tf.matmul(h2, nf['w_ws']) + nf['b_ws']
        bs = tf.matmul(h2, nf['w_bs']) + nf['b_bs']
        self.flow_params = (us, ws, bs)  # #####################
        eps = tf.random_normal(shape=tf.shape(self.z_mean), mean=0, stddev=1)
        z0 = tf.add(self.z_mean, tf.multiply(tf.sqrt(self.z_var), eps))
        if self.flow_type == "Planar":
            # currentClass = NormalizingPlanarFlow(z, self.n_z)
            z_k, sum_log_detj = self.planar_flow_2(z0, self.flow_params, self.nFlows,
                                                       self.n_z, self.invert_condition)
        elif self.flow_type == 'Radial':
            if self.radial_flow_type == "Given in the paper on NF":
                # currentClass = NormalizingRadialFlow(z, self.n_z, self.radial_flow_type)
                z_k, sum_log_detj = self.radial_flow(z0, self.flow_params, self.nFlows,
                                                       self.n_z, self.invert_condition)
        return z0, z_k, sum_log_detj

    def inference_generation_user(self, x):
        with tf.variable_scope("inference_user", reuse=tf.AUTO_REUSE):
            rec_user = {'W1': tf.get_variable("W1", [self.input_dim_user, self.dims[0]],  # W1 = 26x200
                                         initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                   'b1': tf.get_variable("b1", [self.dims[0]],  # b1 = 200
                                         initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                   'W2': tf.get_variable("W2", [self.dims[0], self.dims[1]],  # W2 = 200x100
                                         initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                   'b2': tf.get_variable("b2", [self.dims[1]],  # b2 = 100
                                         initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                   'W_z_mean': tf.get_variable("W_z_mean", [self.dims[1], self.n_z],  # 100x50
                                               initializer=tf.contrib.layers.xavier_initializer(),
                                               dtype=tf.float32),
                   'b_z_mean': tf.get_variable("b_z_mean", [self.n_z],  # 50
                                               initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                   # 'W_z_mean': tf.get_variable("W_z_mean", [self.dims[0], self.n_z],  # 100x50
                   #                             initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                   # 'b_z_mean': tf.get_variable("b_z_mean", [self.n_z],  # 50
                   #                             initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                   # 'W_z_log_sigma': tf.get_variable("W_z_log_sigma", [self.dims[0], self.n_z],  # 100x50
                   #                                  initializer=tf.contrib.layers.xavier_initializer(),
                   #                                  dtype=tf.float32),
                   # 'b_z_log_sigma': tf.get_variable("b_z_log_sigma", [self.n_z],  # 50
                   #                                  initializer=tf.constant_initializer(0.0), dtype=tf.float32)}
                   'W_z_log_sigma': tf.get_variable("W_z_log_sigma", [self.dims[1], self.n_z],  # 100x50
                                                    initializer=tf.contrib.layers.xavier_initializer(),
                                                    dtype=tf.float32),
                   'b_z_log_sigma': tf.get_variable("b_z_log_sigma", [self.n_z],  # 50
                                                    initializer=tf.constant_initializer(0.0), dtype=tf.float32)}
            with tf.variable_scope("normalizing_flow"):
                nf = {'w_us': tf.get_variable("w_us", [self.dims[1], self.nFlows * self.n_z],  # W1 = 8000x200
                                              initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                      'b_us': tf.get_variable("b_us", [self.nFlows * self.n_z],  # b1 = 200
                                              initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                      'w_ws': tf.get_variable("w_ws", [self.dims[1], self.nFlows * self.n_z],  # W2 = 200x100
                                              initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                      'b_ws': tf.get_variable("b_ws", [self.nFlows * self.n_z],  # b2 = 100
                                              initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                      'w_bs': tf.get_variable("w_bs", [self.dims[1], self.n_z],  # 100x50
                                              initializer=tf.contrib.layers.xavier_initializer(),
                                              dtype=tf.float32),
                      'b_bs': tf.get_variable("b_z_mean", [self.n_z],  # 50
                                              initializer=tf.constant_initializer(0.0), dtype=tf.float32)}
        self.weights += [rec_user['W1'], rec_user['b1'], rec_user['W2'], rec_user['b2'], rec_user['W_z_mean'],
                         rec_user['b_z_mean'], rec_user['W_z_log_sigma'], rec_user['b_z_log_sigma']]
        # self.weights += [rec['W1'], rec['b1'], rec['W_z_mean'],
        #                  rec['b_z_mean'], rec['W_z_log_sigma'], rec['b_z_log_sigma']]
        self.reg_loss += tf.nn.l2_loss(rec_user['W1']) + tf.nn.l2_loss(rec_user['W2'])
        # self.reg_loss += tf.nn.l2_loss(rec['W1'])
        # self.x = tf.placeholder(tf.float32, [None, self.input_dim], name='x')
        h1 = tf.nn.sigmoid(
            tf.matmul(x, rec_user['W1']) + rec_user['b1'])  # 16980x200
        # attention_output, alphas = attention(h2, attention_size=50, keep_prob=0.5, return_alphas=True)
        h2 = self.activate(
            tf.matmul(h1, rec_user['W2']) + rec_user['b2'], self.activations[1])  # 16980x100
        h2 = tf.reshape(h2, [-1, 1, self.dims[1]])
        # h1 = tf.reshape(h1, [-1, 1, 100])
        attention_output, alphas = attention(h2, attention_size=50, keep_prob=0.5, time_major=True,
                                             return_alphas=True)
        # attention_output, alphas = attention(h1, attention_size=50, keep_prob=0.5, time_major=True, return_alphas=True)

        self.z_mean_user = tf.matmul(attention_output, rec_user['W_z_mean']) + rec_user['b_z_mean']  # 16980x50#############
        self.z_log_sigma_sq_user = tf.matmul(attention_output, rec_user['W_z_log_sigma']) + rec_user[
            'b_z_log_sigma']  # 16980x50################
        self.z_var_user = tf.exp(self.z_log_sigma_sq)

        # Normalizing Flow parameters
        us_user = tf.matmul(attention_output, nf['w_us']) + nf['b_us']
        ws_user = tf.matmul(attention_output, nf['w_ws']) + nf['b_ws']
        bs_user = tf.matmul(attention_output, nf['w_bs']) + nf['b_bs']
        self.flow_params_user = (us_user, ws_user, bs_user)  # #####################
        eps = tf.random_normal(shape=tf.shape(self.z_mean_user), mean=0, stddev=1)
        z0 = tf.add(self.z_mean_user, tf.multiply(tf.sqrt(self.z_var_user), eps))
        if self.flow_type == "Planar":
            # currentClass = NormalizingPlanarFlow(z, self.n_z)
            z_k, sum_log_detj = self.planar_flow_2(z0, self.flow_params_user, self.nFlows,
                                                   self.n_z, self.invert_condition)
        elif self.flow_type == 'Radial':
            if self.radial_flow_type == "Given in the paper on NF":
                # currentClass = NormalizingRadialFlow(z, self.n_z, self.radial_flow_type)
                z_k, sum_log_detj = self.radial_flow(z0, self.flow_params, self.nFlows,
                                                     self.n_z, self.invert_condition)
        return z0, z_k, sum_log_detj

        #         params.batch_size 128          self.n_z    50
        # eps = tf.random_normal(tf.shape(self.std_q))
        # tf.random_normal()函数用于从服从指定正太分布的数值中取出指定个数的值
        # w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
        # [[-0.81131822  1.48459876  0.06532937]
        # [-2.4427042   0.0992484   0.59122431]]
        # z = self.z_mean + tf.sqrt(tf.maximum(self.std_q, 1e-10)) * eps  # (?, 50)
        # self.z_mean 16980x50
        # return self.z_mean, self.z_log_sigma_sq, lambd

    def generation(self, z_k):
        with tf.variable_scope("generation",reuse=tf.AUTO_REUSE):
            gen = {'w_h': tf.get_variable("w_h", [self.n_z, self.dims[1]],  # 50x100
                                         initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                   'b_h': tf.get_variable("b_h", [self.dims[1]],  # 100
                                         initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                   'w_mu': tf.get_variable("w_mu", [self.dims[1], self.dims[0]],  # W2 = 200x100
                                         initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                   'b_mu': tf.get_variable("b_mu", [self.dims[0]],  # b2 = 100
                                         initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                   'w_v': tf.get_variable("w_v", [self.dims[0], self.input_dim],  # W1 = 8000x200
                                         initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                   'b_v': tf.get_variable("b_v", [self.input_dim],  # b1 = 200
                                         initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                   'w_v1': tf.get_variable("w_v1", [self.dims[0], self.input_dim],  # W1 = 8000x200
                                          initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                   'b_v1': tf.get_variable("b_v1", [self.input_dim],  # b1 = 200
                                          initializer=tf.constant_initializer(0.0), dtype=tf.float32)}


            # gen = {'w_h': tf.get_variable("w_h", [self.n_z, self.dims[0]],  # 50x100
            #                               initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
            #        'b_h': tf.get_variable("b_h", [self.dims[0]],  # 100
            #                               initializer=tf.constant_initializer(0.0), dtype=tf.float32),
            #        'w_mu': tf.get_variable("w_mu", [self.dims[0], self.input_dim],  # W2 = 200x100
            #                                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
            #        'b_mu': tf.get_variable("b_mu", [self.input_dim],  # b2 = 100
            #                                initializer=tf.constant_initializer(0.0), dtype=tf.float32),
            #        # 'w_v': tf.get_variable("w_v", [self.dims[0], self.input_dim],  # W1 = 8000x200
            #        #                        initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
            #        # 'b_v': tf.get_variable("b_v", [self.input_dim],  # b1 = 200
            #        #                        initializer=tf.constant_initializer(0.0), dtype=tf.float32),
            #        'w_v1': tf.get_variable("w_v1", [self.dims[0], self.input_dim],  # W1 = 8000x200
            #                                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
            #        'b_v1': tf.get_variable("b_v1", [self.input_dim],  # b1 = 200
            #                                initializer=tf.constant_initializer(0.0), dtype=tf.float32)
            #        }

        # self.weights += [gen['w_h'], gen['b_h'], gen['b_v1']]
        # self.reg_loss += tf.nn.l2_loss(gen['w_mu']) + tf.nn.l2_loss(gen['w_v1'])
        h2 = self.activate(
            tf.matmul(z_k, gen['w_h']) + gen['b_h'], self.activations[1])
        h1 = tf.nn.relu(
            tf.matmul(h2, gen['w_mu']) + gen['b_mu'])
        out_mu = tf.matmul(h1, gen['w_v']) + gen['b_v']

        out_log_var = tf.matmul(h1, gen['w_v1']) + gen['b_v1']
        out = tf.nn.sigmoid(out_mu)
        return out, out_mu, out_log_var  # 128X8000这一部分是用来干嘛的？

    def generation_user(self, z_k):
        with tf.variable_scope("generation_user", reuse=tf.AUTO_REUSE):
            gen_user = {'w_h': tf.get_variable("w_h", [self.n_z, self.dims[1]],  # 50x100
                                         initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                   'b_h': tf.get_variable("b_h", [self.dims[1]],  # 100
                                         initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                   'w_mu': tf.get_variable("w_mu", [self.dims[1], self.dims[0]],  # W2 = 200x100
                                         initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                   'b_mu': tf.get_variable("b_mu", [self.dims[0]],  # b2 = 100
                                         initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                   'w_v': tf.get_variable("w_v", [self.dims[0], self.input_dim_user],  # W1 = 8000x200
                                         initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                   'b_v': tf.get_variable("b_v", [self.input_dim_user],  # b1 = 200
                                         initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                   'w_v1': tf.get_variable("w_v1", [self.dims[0], self.input_dim_user],  # W1 = 8000x200
                                          initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                   'b_v1': tf.get_variable("b_v1", [self.input_dim_user],  # b1 = 200
                                          initializer=tf.constant_initializer(0.0), dtype=tf.float32)}


            # gen = {'w_h': tf.get_variable("w_h", [self.n_z, self.dims[0]],  # 50x100
            #                               initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
            #        'b_h': tf.get_variable("b_h", [self.dims[0]],  # 100
            #                               initializer=tf.constant_initializer(0.0), dtype=tf.float32),
            #        'w_mu': tf.get_variable("w_mu", [self.dims[0], self.input_dim],  # W2 = 200x100
            #                                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
            #        'b_mu': tf.get_variable("b_mu", [self.input_dim],  # b2 = 100
            #                                initializer=tf.constant_initializer(0.0), dtype=tf.float32),
            #        # 'w_v': tf.get_variable("w_v", [self.dims[0], self.input_dim],  # W1 = 8000x200
            #        #                        initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
            #        # 'b_v': tf.get_variable("b_v", [self.input_dim],  # b1 = 200
            #        #                        initializer=tf.constant_initializer(0.0), dtype=tf.float32),
            #        'w_v1': tf.get_variable("w_v1", [self.dims[0], self.input_dim],  # W1 = 8000x200
            #                                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
            #        'b_v1': tf.get_variable("b_v1", [self.input_dim],  # b1 = 200
            #                                initializer=tf.constant_initializer(0.0), dtype=tf.float32)
            #        }

        # self.weights += [gen_user['w_h'], gen_user['b_h'], gen_user['b_v1']]
        # self.reg_loss += tf.nn.l2_loss(gen_user['w_mu']) + tf.nn.l2_loss(gen_user['w_v1'])
        h2 = self.activate(
            tf.matmul(z_k, gen_user['w_h']) + gen_user['b_h'], self.activations[1])
        h1 = tf.nn.relu(
            tf.matmul(h2, gen_user['w_mu']) + gen_user['b_mu'])
        out_mu = tf.matmul(h1, gen_user['w_v']) + gen_user['b_v']

        out_log_var = tf.matmul(h1, gen_user['w_v1']) + gen_user['b_v1']
        out = tf.nn.sigmoid(out_mu)
        return out, out_mu, out_log_var  # 128X8000这一部分是用来干嘛的？

    def cdl_estimate(self, data_x, data_x_user, num_iter, user_ratings, user_ratings_test, item_count):
        loss_app = []
        flow_loss_app = []
        vloss_app = []
        bprloss_app = []
        # z_k_app =[]
        # z_k_user_app = []
        for i in range(num_iter):
            uij = self.generate_train_batch(user_ratings, user_ratings_test, item_count)
            # _bprloss, _train_op = self.sess.run([bprloss, train_op],
            #                                   feed_dict={self.u: uij[:, 0], self.i: uij[:, 1], self.j: uij[:, 2]})
            # _batch_bprloss += _bprloss
            b_x, ids = utils.get_batch(data_x, self.params.batch_size)  # 从data_x中随机选择128个item的8000表示
            b_x_user, ids_user = utils.get_batch(data_x_user, self.params.batch_size)
            _, all_loss, flow_loss, flow_user_loss, v_loss, recons_loss1, kl_loss1, z_k, z_k_user, z0, z0_user = \
                                    self.sess.run((self.optimizer, self.loss, self.loss_op, self.loss_op_user,
                                                   self.v_loss,
                                                   self.recons_loss1, self.kl_loss1, self.z_k,
                                                   self.z_k_user, self.z0, self.z_user),
                                                                    feed_dict={self.x: b_x, self.v: self.m_V[ids, :],
                                                                               self.x_user: b_x_user,
                                                                               self.v_user: self.m_U[ids_user, :]})
            if i == 0:
                z_k_cat = z_k
                z_k_user_cat = z_k_user
                z0_cat = z0
                z0_user_cat = z0_user
            else:
                z_k_cat = np.concatenate((z_k_cat, z_k), axis=0)
                z_k_user_cat = np.concatenate((z_k_user_cat, z_k_user), axis=0)
                z0_cat = np.concatenate((z0_cat, z0), axis=0)
                z0_user_cat = np.concatenate((z0_user_cat, z0_user), axis=0)
            # z_k = np.array(z_k)
            # z_k_app.append(z_k)
            # z_k_user = np.array(z_k_user)
            # z_k_user_app.append(z_k_user)
            # # Display logs per epoch step
            # print 'recons_loss1: ', recons_loss1
            # print 'kl_loss1:', kl_loss1
            # kl_loss1: -1293.14
            # recons_loss1:  4194.75
            # kl_loss1: -703.764
            if i % self.print_step == 0:
                print "cdl_estimate_Iter:", '%04d' % (i + 1), \
                    "loss=", "{:.5f}".format(all_loss), \
                    "flowloss=", "{:.5f}".format(flow_loss), \
                    "vloss=", "{:.5f}".format(v_loss)

            loss_app.append(all_loss)
            flow_loss_app.append(flow_loss)
            vloss_app.append(v_loss)

            user_count = 0
            _auc_sum = 0.0

            # each batch will return only one user's auc
            # AUC: 从每个用户的(u, i, j)偏序关系中抽出一条用作测试。
            # for t_uij in self.generate_test_batch(user_ratings, user_ratings_test, item_count):
            #     _auc, test_bprloss_result = self.sess.run([self.mf_auc, self.bprloss],
            #                                                feed_dict={self.u: t_uij[:, 0], self.i: t_uij[:, 1],
            #                                                           self.j: t_uij[:, 2]})
            #     user_count += 1
            #     _auc_sum += _auc
            # print ("test_loss: ", test_bprloss_result, "test_auc: ", _auc_sum / user_count)
            # print ("")

        print 'loss_one_eppoch: ', np.mean(loss_app), 'flowloss_one_eppoch: ', np.mean(flow_loss_app), \
            'vloss_one_eppoch: ', np.mean(vloss_app)
        z_k_cat = np.array(z_k_cat)
        z_k_user_cat = np.array(z_k_user_cat)
        z0_cat = np.array(z0_cat)
        z0_user_cat = np.array(z0_user_cat)
        return flow_loss, flow_user_loss, z_k_cat, z_k_user_cat, z0_cat, z0_user_cat

    def transform(self, data_x):
        data_en = self.sess.run(self.z_mean, feed_dict={self.x: data_x})
        return data_en
    def transform_user(self, data_x):
        data_en = self.sess.run(self.z_mean_user, feed_dict={self.x_user: data_x})
        return data_en

    def pmf_estimate(self, users, items, test_users, test_items, params):
        """
        users: list of list  users指的是每个用户访问的点的个数，items指的是每个item被几个用户访问过。
        """
        min_iter = 1
        a_minus_b = params.a - params.b  # 1 - 0.01
        converge = 1.0
        likelihood_old = 0.0
        likelihood = -math.exp(20)  # -485165195.41
        it = 0
        while ((it < params.max_iter and converge > 1e-6) or it < min_iter):
            likelihood_old = likelihood
            likelihood = 0
            # update U  items存放每一个item被哪一些用户访问
            # VV^T for v_j that has at least one user liked
            ids = np.array([len(x) for x in items]) > 0  # ids是一个布尔值的数组，item被用户访问的次数大于1的为True,否则为Flase.
            v = self.m_V[ids]  # 总共16980个items，只有15246个items是被至少一个用户访问过的。（15246, 50）
            VVT = np.dot(v.T, v)  # （50, 50）
            XX = VVT * params.b + np.eye(self.m_num_factors) * params.lambda_u  # （50, 50）
            #                (50, 50)*0.01  + (50,50) * 0.1
            for i in xrange(self.m_num_users):
                item_ids = users[i]  # 每个用户访问过的items
                n = len(item_ids)  # 训练集中每个用户访问的items数
                if n > 0:
                    A = np.copy(XX)  # 27.3396417505
                    A += np.dot(self.m_V[item_ids, :].T, self.m_V[item_ids, :]) * a_minus_b  # a_minus_b = 0.99
                    B = np.copy(A)
                    A += np.eye(self.m_num_factors) * params.lambda_u
                    x = params.a * np.sum(self.m_V[item_ids, :], axis=0) + params.lambda_u * self.m_theta_user[i, :]
                    # （50, ）self.m_V[item_ids, :] = (10, 50) 按列相加
                    self.m_U[i, :] = scipy.linalg.solve(A, x)  # 求解线性矩阵方程或线性标量方程组 A * self.m_U[i, :] = x
                    #                   已知A和x求 self.m_U[i, :]的值,
                    likelihood += -0.5 * n * params.a
                    likelihood += params.a * np.sum(np.dot(self.m_V[item_ids, :], self.m_U[i, :][:, np.newaxis]),
                                                    axis=0)
                    likelihood += -0.5 * self.m_U[i, :].dot(B).dot(self.m_U[i, :][:, np.newaxis])
                    # likelihood += -0.5 * params.lambda_u * np.sum(self.m_U[i] * self.m_U[i])
                    ep_user = self.m_U[i, :] - self.m_theta_user[i, :]
                    likelihood += -0.5 * params.lambda_v * np.sum(ep_user * ep_user)
                else:
                    # m=0, this article has never been rated
                    A = np.copy(XX)
                    A += np.eye(self.m_num_factors) * params.lambda_v
                    x = params.lambda_v * self.m_theta_user[i, :]
                    self.m_U[i, :] = scipy.linalg.solve(A, x)

                    ep_user = self.m_U[i, :] - self.m_theta_user[i, :]
                    likelihood += -0.5 * params.lambda_v * np.sum(ep_user * ep_user)
                    # 考虑ULE三个部分的相乘，加geo和social
                    #                                 0.5  *  0.1            *
            # update V
            ids = np.array([len(x) for x in users]) > 0  # 用户访问的items的个数
            u = self.m_U[ids]  # 所有用户访问的items个数都是10个，所以是（5551, 50）
            XX = np.dot(u.T, u) * params.b  # （50, 50） × 0.01   0.51677043......
            for j in xrange(self.m_num_items):
                user_ids = items[j]
                m = len(user_ids)
                if m > 0:
                    A = np.copy(XX)  # XX就是每个用户访问的
                    A += np.dot(self.m_U[user_ids, :].T, self.m_U[user_ids, :]) * a_minus_b
                    B = np.copy(A)  # 0.52194386
                    A += np.eye(self.m_num_factors) * params.lambda_v  # params.lambda_v = 10
                    x = params.a * np.sum(self.m_U[user_ids, :], axis=0) + params.lambda_v * self.m_theta[j, :]
                    self.m_V[j, :] = scipy.linalg.solve(A, x)  # params.a = 1

                    likelihood += -0.5 * m * params.a  # -0.5 * 某个item被多少个用户访问 * 1
                    likelihood += params.a * np.sum(np.dot(self.m_U[user_ids, :], self.m_V[j, :][:, np.newaxis]),
                                                    axis=0)
                    likelihood += -0.5 * self.m_V[j, :].dot(B).dot(self.m_V[j, :][:, np.newaxis])

                    ep = self.m_V[j, :] - self.m_theta[j, :]
                    likelihood += -0.5 * params.lambda_v * np.sum(ep * ep)
                else:
                    # m=0, this article has never been rated
                    A = np.copy(XX)
                    A += np.eye(self.m_num_factors) * params.lambda_v
                    x = params.lambda_v * self.m_theta[j, :]
                    self.m_V[j, :] = scipy.linalg.solve(A, x)

                    ep = self.m_V[j, :] - self.m_theta[j, :]
                    likelihood += -0.5 * params.lambda_v * np.sum(ep * ep)
            # computing negative log likelihood
            # likelihood += -0.5 * params.lambda_u * np.sum(self.m_U * self.m_U)
            # likelihood += -0.5 * params.lambda_v * np.sum(self.m_V * self.m_V)
            # split R_ij into 0 and 1
            # -sum(0.5*C_ij*(R_ij - u_i^T * v_j)^2) = -sum_ij 1(R_ij=1) 0.5*C_ij +
            #  sum_ij 1(R_ij=1) C_ij*u_i^T * v_j - 0.5 * sum_j v_j^T * U C_i U^T * v_j

            it += 1
            converge = abs(
                1.0 * (likelihood - likelihood_old) / likelihood_old)  # 第一轮的likelihood_old = -485165195.41

            if self.verbose:
                if likelihood < likelihood_old:
                    print("likelihood is decreasing!")

                print("[iter=%04d], likelihood=%.5f, converge=%.10f" % (it, likelihood, converge))

        return likelihood

    def activate(self, linear, name):
        if name == 'sigmoid':
            return tf.nn.sigmoid(linear, name='encoded')
        elif name == 'softmax':
            return tf.nn.softmax(linear, name='encoded')
        elif name == 'linear':
            return linear
        elif name == 'tanh':
            return tf.nn.tanh(linear, name='encoded')
        elif name == 'relu':
            return tf.nn.relu(linear, name='encoded')

    def run(self, users, items, test_users, test_items, data_x, data_x_user, user_ratings, user_ratings_test, params):

    # def run(self, users, items, test_users, test_items, data_x, params):

        # model.run(data["train_users"], data["train_items"], data["test_users"], data["test_items"],
        #           data["content"], params)data_x是一个16980x8000的矩阵
        self.m_theta[:] = self.transform(data_x)  # 16980X50
        self.m_V[:] = self.m_theta  # 16980X50

        self.m_theta_user[:] = self.transform_user(data_x_user)  # 16980X50
        self.m_U[:] = self.m_theta_user  # 16980X50
        #   m_theta的值此时与m_V的一样。self.m_theta = 0.1 * np.random.randn(self.m_num_items, self.m_num_factors)#16980x50
        n = data_x.shape[0]  # 16980

        n_user = data_x_user.shape[0]
        for epoch in range(100):
            # num_iter = int(1)
            num_iter = int(n / params.batch_size)
            # gen_loss = self.cdl_estimate(data_x, params.cdl_max_iter)

            gen_loss, flow_user_loss, z_k, z_k_user, z0, z0_user =\
                self.cdl_estimate(data_x, data_x_user, num_iter, user_ratings, user_ratings_test, n)
            # gen_loss = self.cdl_estimate(data_x, num_iter)
            self.m_theta[:] = self.transform(data_x)
            likelihood = self.pmf_estimate(users, items, test_users, test_items, params)
            loss = -likelihood + 0.25 * gen_loss * n * params.lambda_r + 0.25 * flow_user_loss * n_user * params.lambda_r
            logging.info("momo[#epoch=%06d], loss=%.5f, neg_likelihood=%.5f, gen_loss=%.5f" % (
                epoch, loss, -likelihood, gen_loss))
        # model_epoch_test.run(data["train_users"], data["train_items"], data["test_users"], data["test_items"],
        #           data["content"], params)data_x是一个16980x8000的矩阵
        # self.m_theta_user[:] = self.transform(data_x)  # 16980X50
        # self.m_U[:] = self.m_theta_user  # 16980X50
        # # self.m_theta = 0.1 * np.random.randn(self.m_num_items, self.m_num_factors)#16980x50
        # n = data_x.shape[0]  # 16980

        # for epoch in range(60):
        # # for epoch in range(params.n_epochs):
        #     num_iter = int(n / params.batch_size)
        #     gen_loss = self.cdl_estimate(data_x, num_iter, user_ratings, user_ratings_test, n)
        #     # values = sess.run(variable_names)
        #     self.m_theta_user[:] = self.transform(data_x)
        #     likelihood = self.pmf_estimate(users, items, test_users, test_items, params)
        #     loss = -likelihood + 0.5 * gen_loss * n * params.lambda_r
        #     logging.info("momo[#epoch=%06d], loss=%.5f, neg_likelihood=%.5f, gen_loss=%.5f" % (
        #         epoch, loss, -likelihood, gen_loss))

            if epoch == 25:
                self.save_model(
                    weight_path="model_1_flow/cvae_flow25",
                    pmf_path="model_1_flow/pmf_flow25")
                np.savetxt('/home/yinruiyang/Pycharm_Proj_Mo/Rectify_CVAE/data/MovieLens/Resource/code/model_1_flow/z_k25.txt', z_k)
                np.savetxt('/home/yinruiyang/Pycharm_Proj_Mo/Rectify_CVAE/data/MovieLens/Resource/code/model_1_flow/z_k_user25.txt', z_k_user)
                np.savetxt(
                    '/home/yinruiyang/Pycharm_Proj_Mo/Rectify_CVAE/data/MovieLens/Resource/code/model_1_flow/z0_25.txt',
                    z0)
                np.savetxt(
                    '/home/yinruiyang/Pycharm_Proj_Mo/Rectify_CVAE/data/MovieLens/Resource/code/model_1_flow/z0_user25.txt',
                    z0_user)
                # self.save_model(
                #     weight_path="model_z/cvae_z",
                #     pmf_path="model_z/pmf_z")
        ##################################################
            if epoch == 10:
                self.save_model(
                    weight_path="model_1_flow/cvae_flow10",
                    pmf_path="model_1_flow/pmf_flow10")
                np.savetxt(
                    '/home/yinruiyang/Pycharm_Proj_Mo/Rectify_CVAE/data/MovieLens/Resource/code/model_1_flow/z_k10.txt',
                    z_k)
                np.savetxt(
                    '/home/yinruiyang/Pycharm_Proj_Mo/Rectify_CVAE/data/MovieLens/Resource/code/model_1_flow/z_k_user10.txt',
                    z_k_user)
                np.savetxt(
                    '/home/yinruiyang/Pycharm_Proj_Mo/Rectify_CVAE/data/MovieLens/Resource/code/model_1_flow/z0_10.txt',
                    z0)
                np.savetxt(
                    '/home/yinruiyang/Pycharm_Proj_Mo/Rectify_CVAE/data/MovieLens/Resource/code/model_1_flow/z0_user10.txt',
                    z0_user)
                ##################################################
            if epoch == 20:
                self.save_model(
                    weight_path="model_1_flow/cvae_flow20",
                    pmf_path="model_1_flow/pmf_flow20")
                np.savetxt(
                    '/home/yinruiyang/Pycharm_Proj_Mo/Rectify_CVAE/data/MovieLens/Resource/code/model_1_flow/z_k20.txt',
                    z_k)
                np.savetxt(
                    '/home/yinruiyang/Pycharm_Proj_Mo/Rectify_CVAE/data/MovieLens/Resource/code/model_1_flow/z_k_user20.txt',
                    z_k_user)
                np.savetxt(
                    '/home/yinruiyang/Pycharm_Proj_Mo/Rectify_CVAE/data/MovieLens/Resource/code/model_1_flow/z0_20.txt',
                    z0)
                np.savetxt(
                    '/home/yinruiyang/Pycharm_Proj_Mo/Rectify_CVAE/data/MovieLens/Resource/code/model_1_flow/z0_user20.txt',
                    z0_user)
            if epoch == 30:
                self.save_model(
                    weight_path="model_1_flow/cvae_flow25",
                    pmf_path="model_1_flow/pmf_flow25")
                np.savetxt('/home/yinruiyang/Pycharm_Proj_Mo/Rectify_CVAE/data/MovieLens/Resource/code/model_1_flow/z_k30.txt', z_k)
                np.savetxt('/home/yinruiyang/Pycharm_Proj_Mo/Rectify_CVAE/data/MovieLens/Resource/code/model_1_flow/z_k_user30.txt', z_k_user)
                np.savetxt(
                    '/home/yinruiyang/Pycharm_Proj_Mo/Rectify_CVAE/data/MovieLens/Resource/code/model_1_flow/z0_30.txt',
                    z0)
                np.savetxt(
                    '/home/yinruiyang/Pycharm_Proj_Mo/Rectify_CVAE/data/MovieLens/Resource/code/model_1_flow/z0_user30.txt',
                    z0_user)
                # self.save_model(
                #     weight_path="model_z/cvae_z",
                #     pmf_path="model_z/pmf_z")
        ##################################################
            if epoch == 40:
                self.save_model(
                    weight_path="model_1_flow/cvae_flow40",
                    pmf_path="model_1_flow/pmf_flow40")
                np.savetxt(
                    '/home/yinruiyang/Pycharm_Proj_Mo/Rectify_CVAE/data/MovieLens/Resource/code/model_1_flow/z_k40.txt',
                    z_k)
                np.savetxt(
                    '/home/yinruiyang/Pycharm_Proj_Mo/Rectify_CVAE/data/MovieLens/Resource/code/model_1_flow/z_k_user40.txt',
                    z_k_user)
                np.savetxt(
                    '/home/yinruiyang/Pycharm_Proj_Mo/Rectify_CVAE/data/MovieLens/Resource/code/model_1_flow/z0_40.txt',
                    z0)
                np.savetxt(
                    '/home/yinruiyang/Pycharm_Proj_Mo/Rectify_CVAE/data/MovieLens/Resource/code/model_1_flow/z0_user40.txt',
                    z0_user)
                ##################################################
            if epoch == 50:
                self.save_model(
                    weight_path="model_1_flow/cvae_flow50",
                    pmf_path="model_1_flow/pmf_flow50")
                np.savetxt(
                    '/home/yinruiyang/Pycharm_Proj_Mo/Rectify_CVAE/data/MovieLens/Resource/code/model_1_flow/z_k50.txt',
                    z_k)
                np.savetxt(
                    '/home/yinruiyang/Pycharm_Proj_Mo/Rectify_CVAE/data/MovieLens/Resource/code/model_1_flow/z_k_user50.txt',z_k_user)

                np.savetxt(
                    '/home/yinruiyang/Pycharm_Proj_Mo/Rectify_CVAE/data/MovieLens/Resource/code/model_1_flow/z0_50.txt',
                    z0)
                np.savetxt(
                    '/home/yinruiyang/Pycharm_Proj_Mo/Rectify_CVAE/data/MovieLens/Resource/code/model_1_flow/z0_user50.txt',
                    z0_user)

        #     if epoch == 60:
        #         self.save_model(
        #             weight_path="model_1_flow/cvae_flow60",
        #             pmf_path="model_1_flow/pmf_flow60")
        #         np.savetxt('/home/yinruiyang/Pycharm_Proj_Mo/Rectify_CVAE/data/MovieLens/Resource/code/model_1_flow/z_k60.txt', z_k)
        #         np.savetxt('/home/yinruiyang/Pycharm_Proj_Mo/Rectify_CVAE/data/MovieLens/Resource/code/model_1_flow/z_k_user60.txt', z_k_user)
        #         # self.save_model(
        #         #     weight_path="model_z/cvae_z",
        #         #     pmf_path="model_z/pmf_z")
        # ##################################################
        #     if epoch == 70:
        #         self.save_model(
        #             weight_path="model_1_flow/cvae_flow70",
        #             pmf_path="model_1_flow/pmf_flow70")
        #         np.savetxt(
        #             '/home/yinruiyang/Pycharm_Proj_Mo/Rectify_CVAE/data/MovieLens/Resource/code/model_1_flow/z_k70.txt',
        #             z_k)
        #         np.savetxt(
        #             '/home/yinruiyang/Pycharm_Proj_Mo/Rectify_CVAE/data/MovieLens/Resource/code/model_1_flow/z_k_user70.txt',
        #             z_k_user)
        #         ##################################################
        #     if epoch == 80:
        #         self.save_model(
        #             weight_path="model_1_flow/cvae_flow80",
        #             pmf_path="model_1_flow/pmf_flow80")
        #         np.savetxt(
        #             '/home/yinruiyang/Pycharm_Proj_Mo/Rectify_CVAE/data/MovieLens/Resource/code/model_1_flow/z_k80.txt',
        #             z_k)
        #         np.savetxt(
        #             '/home/yinruiyang/Pycharm_Proj_Mo/Rectify_CVAE/data/MovieLens/Resource/code/model_1_flow/z_k_user80.txt',z_k_user)
        #     if epoch == 90:
        #         self.save_model(
        #             weight_path="model_1_flow/cvae_flow90",
        #             pmf_path="model_1_flow/pmf_flow90")
        #         np.savetxt(
        #             '/home/yinruiyang/Pycharm_Proj_Mo/Rectify_CVAE/data/MovieLens/Resource/code/model_1_flow/z_k90.txt',
        #             z_k)
        #         np.savetxt(
        #             '/home/yinruiyang/Pycharm_Proj_Mo/Rectify_CVAE/data/MovieLens/Resource/code/model_1_flow/z_k_user90.txt',z_k_user)


        np.savetxt('/home/yinruiyang/Pycharm_Proj_Mo/Rectify_CVAE/data/MovieLens/Resource/code/model_1_flow/z_k100.txt', z_k)
        np.savetxt('/home/yinruiyang/Pycharm_Proj_Mo/Rectify_CVAE/data/MovieLens/Resource/code/model_1_flow/z_k_user100.txt',
                           z_k_user)
    def save_model(self, weight_path, pmf_path=None):
        self.saver.save(self.sess, weight_path)
        logging.info("Weights saved at " + weight_path)
        if pmf_path is not None:
            scipy.io.savemat(pmf_path, {"m_U": self.m_U, "m_V": self.m_V, "m_theta": self.m_theta})
            logging.info("Weights saved at " + pmf_path)
            # model_epoch_test.load_model(weight_path="model_epoch_test/pretrain")

    def save_model1(self, weight_path, pmf_path=None):
        self.saver.save(self.sess, weight_path)
        logging.info("Weights saved at " + weight_path)
        if pmf_path is not None:
            scipy.io.savemat(pmf_path, {"m_U": self.m_U, "m_V": self.m_V, "m_theta": self.m_theta})
            logging.info("Weights saved at " + pmf_path)
            # model_epoch_test.load_model(weight_path="model_epoch_test/pretrain")

    def load_model(self, weight_path, pmf_path=None):
        logging.info("Loading weights from " + weight_path)
        self.saver.restore(self.sess, weight_path)
        if pmf_path is not None:
            logging.info("Loading pmf data from " + pmf_path)
            data = scipy.io.loadmat(pmf_path)
            self.m_U[:] = data["m_U"]
            self.m_V[:] = data["m_V"]
            self.m_theta[:] = data["m_theta"]

    # #########BPR###############################
    def generate_train_batch(self, user_ratings, user_ratings_test, item_count, batch_size=1000):
        t = []
        for b in range(batch_size):
            u = random.sample(user_ratings.keys(), 1)[0]
            i = random.sample(user_ratings[u], 1)[0]
            while i == user_ratings_test[u]:
                i = random.sample(user_ratings[u], 1)[0]

            j = random.randint(1, item_count)
            while j in user_ratings[u]:
                j = random.randint(1, item_count)
            t.append([u, i, j])
        return np.asarray(t)

    def generate_test_batch(self, user_ratings, user_ratings_test, item_count):
        for u in user_ratings.keys():
            t = []
            i = user_ratings_test[u]
            for j in range(0, item_count):
                if not (j in user_ratings[u]):
                    t.append([u, i, j])
            yield np.asarray(t)


    def cross_entropy_loss(self, prediction, actual, offset=1e-4):
        """
        param prediction: tensor of observed values
        param actual: tensor of actual values
        """
        with tf.name_scope("cross_entropy"):
            #        predicate = tf.logical_or(tf.less(1e-10 + prediction, 1e-7),
            #                                  tf.less(1e-10 + 1 + prediction, 1e-7))
            #        fn1 = - tf.reduce_sum(actual * tf.log(1e-8)
            #                         + (1 - actual) * tf.log(1e-8), 1)
            #        fn2 = - tf.reduce_sum(actual * tf.log(1e-10 + prediction)
            #                             + (1 - actual) * tf.log(1e-10 + 1 - prediction), 1)
            #        ce_loss = tf.cond(predicate, lambda: fn1, lambda: fn2)
            _prediction = tf.clip_by_value(prediction, offset, 1 - offset)
            ce_loss = - tf.reduce_sum(actual * tf.log(_prediction)
                                      + (1 - actual) * tf.log(1 - _prediction), 1)
            #        ce_loss= - tf.reduce_sum(actual * tf.log(1e-10 + prediction)
            #                             + (1 - actual) * tf.log(1e-10 + 1 - prediction), 1)
            return ce_loss


    def kl_divergence_gaussian(self, mu, var):
        with tf.name_scope("kl_divergence"):
            # kl = - 0.5 * tf.reduce_sum(1.0 + log_var - tf.square(mu) - tf.exp(log_var), 1)
            _var = tf.clip_by_value(var, 1e-4, 1e6)
            kl = - 0.5 * tf.reduce_sum(1 + tf.log(_var) - tf.square(mu) - \
                                       tf.exp(tf.log(_var)), 1)
            return kl


    def gaussian_log_pdf(self, z, mu, var):
        """
        Log probability from a diagonal covariance normal distribution.
        """
        return tf.contrib.distributions.MultivariateNormalDiag(
            loc=mu, scale_diag=tf.maximum(tf.sqrt(var), 1e-4)).log_prob(z + 1e-4)


    def elbo_loss(self, actual, prediction, beta, global_step, z_mu, z_var, z0, zk, sum_log_detj):
        # 实值， 预测值，True, global是变量，z_mu是vae的均值，z_var进行tf.exp之后的方差，encoder之后的z0,zk是flow之后的zk, flow出来的值
        # global_step = tf.Variable(0, trainable=False)
        #    monitor = {}
        mu = z_mu
        _var = z_var

        # First term is +ve. Rest all terms are negative.
        z0 = z0
        zk = zk
        logdet_jacobian = sum_log_detj

        # First term
        log_q0_z0 = self.gaussian_log_pdf(z0, mu, _var)
        # Third term
        # sum_logdet_jacobian = tf.reduce_mean(logdet_jacobian,
        #                                     name='sum_logdet_jacobian')
        sum_logdet_jacobian = logdet_jacobian
        # First term - Third term
        log_qk_zk = log_q0_z0 - sum_logdet_jacobian

        # First component of the second term: p(x|z_k)
        if beta:
            beta_t = tf.minimum(1.0, 0.01 + tf.cast(global_step / 10000, tf.float32))  # global_step
            print 'global_step: ', global_step, 'beta_t: ', beta_t
            log_p_x_given_zk = beta_t * self.cross_entropy_loss(prediction, actual)
            log_p_zk = beta_t * self.gaussian_log_pdf(zk, tf.zeros_like(mu), tf.ones_like(mu))
        else:
            log_p_x_given_zk = self.cross_entropy_loss(prediction, actual)
            log_p_zk = self.gaussian_log_pdf(zk, tf.zeros_like(mu), tf.ones_like(mu))

        # self.recons_loss = log_p_x_given_zk
        # self.kl_loss = log_qk_zk - log_p_zk
        # self._elbo_loss = tf.reduce_mean(self.kl_loss + self.recons_loss)
        recons_loss = log_p_x_given_zk
        kl_loss = log_qk_zk - log_p_zk

        recons_loss1 = tf.reduce_mean(log_p_x_given_zk)
        kl_loss1 = tf.reduce_mean(log_qk_zk - log_p_zk)
        print 'recons_loss1: ', recons_loss1
        print 'kl_loss1:', kl_loss1
        _elbo_loss = tf.reduce_mean(kl_loss + recons_loss)
        # return recons_loss, kl_loss, _elbo_loss
        return recons_loss1, kl_loss1, _elbo_loss

    def make_loss(self, pred, actual, log_var, mu, log_detj, z0, sigma=1.0):
        """
        NOT USING
        """
        kl = -tf.reduce_mean(0.5 * tf.reduce_sum(1.0 + log_var - tf.square(mu) - tf.exp(log_var), 1))
        offset = 1e-7
        prediction_ = tf.clip_by_value(pred, offset, 1 - offset)
        cross_entropy_loss = tf.reduce_mean(actual * tf.log(prediction_) + (1 - actual) * tf.log(1 - prediction_), 1)
        # rec_err = 0.5*(tf.nn.l2_loss(actual - pred)) / sigma
        loss = tf.reduce_mean(kl + cross_entropy_loss - log_detj)
        return loss


    def vanilla_vae_loss(self, x, x_reconstr_mean, z_mu, z_var):
        reconstr_loss = -tf.reduce_sum(x * tf.log(1e-10 + x_reconstr_mean)
                                       + (1 - x) * tf.log(1e-10 + 1 - x_reconstr_mean), 1)
        latent_loss = -0.5 * tf.reduce_sum(1 + tf.log(z_var) - tf.square(z_mu) - \
                                           tf.exp(tf.log(z_var)), 1)
        cost = tf.reduce_mean(reconstr_loss + latent_loss)
        return cost


    def log_normal(self, x, mean, var, eps=1e-5):
        const = - 0.5 * tf.log(2 * math.pi)
        var += eps
        return const - tf.log(var) / 2 - (x - mean) ** 2 / (2 * var)

    def planar_flow_2(self, z, flow_params, nFlows, z_dim,
                      invert_condition=True):
        K = nFlows
        Z = z_dim
        us, ws, bs = flow_params

        log_detjs = []
        if K == 0:
            sum_logdet_jacobian = logdet_jacobian = 0
        else:
            for k in range(K):
                u, w, b = us[:, k * Z:(k + 1) * Z], ws[:, k * Z:(k + 1) * Z], bs[:, k]
                print "u shape", u.get_shape()
                print "w shape", w.get_shape()
                # print "z shape", z.get_shape()
                print "b shape", b.get_shape()
                if invert_condition:
                    uw = tf.reduce_sum(tf.matmul(w, u, transpose_a=True),
                                       axis=1, keep_dims=True)  # u: (?,2), w: (?,2), b: (?,)
                    muw = -1 + tf.nn.softplus(uw)  # = -1 + T.log(1 + T.exp(uw))
                    u_hat = u + tf.multiply(tf.transpose((muw - uw)), w) / tf.norm(w, axis=[-2, -1])
                    print "norm_w shape", tf.norm(w, axis=[-2, -1]).get_shape()
                    print "uw shape", uw.get_shape()
                    print "muw shape", muw.get_shape()
                else:
                    u_hat = u
                print "u_hat shape", u_hat.get_shape()
                zw = tf.reduce_sum(tf.multiply(tf.cast(z, tf.float32), w), axis=1)
                zwb = zw + b
                z = z + u_hat * tf.reshape(tf.tanh(zwb), [-1, 1])  # z is (?,2)
                psi = tf.reshape((1 - tf.tanh(zwb) ** 2), [-1, 1]) * w  # Equation 11. # tanh(x)dx = 1 - tanh(x)**2
                # psi= tf.reduce_sum(tf.matmul(tf.transpose(1-self.tanh(zwb)**2), self.w))
                psi_u = tf.reduce_sum(tf.matmul(u_hat, psi, transpose_b=True),
                                      axis=1, keep_dims=True)
                # psi_u= tf.matmul(tf.transpose(u_hat), tf.transpose(psi)) # Second term in equation 12. u_transpose*psi_z
                logdet_jacobian = tf.log(tf.clip_by_value(tf.abs(1 + psi_u), 1e-4, 1e7))  # Equation 12
                # print "f_z shape", f_z.get_shape()
                log_detjs.append(logdet_jacobian)
                logdet_jacobian = tf.concat(log_detjs[0:nFlows + 1], axis=1)
                sum_logdet_jacobian = tf.reduce_mean(logdet_jacobian)
        return z, sum_logdet_jacobian

    def radial_flow(self, z, flow_params, K, Z, invert_condition=True):
        z0s, alphas, betas = flow_params
        log_detjs = []
        if K == 0:
            sum_logdet_jacobian = logdet_jacobian = 0
        else:
            for k in range(K):
                # z0, alpha, beta = z0s[:, k*Z:(k+1)*Z], alphas[:, k*Z:(k+1)*Z], betas[:, k]
                z0, alpha, beta = z0s[:, k * Z:(k + 1) * Z], alphas[:, k], betas[:, k]
                print "z0 shape", z0.get_shape()
                print "alpha shape", alpha.get_shape()
                print "beta shape", beta.get_shape()
                if invert_condition:
                    m_of_beta = tf.nn.softplus(
                        beta)  # m(x)= log(1 + exp(x)) where x= w'*u. Last equation in A.2 Radial Flows.
                    print "m_of_beta", m_of_beta.get_shape()
                    print "alpha", alpha.get_shape()
                    beta_hat = -alpha + m_of_beta  # It's a scalar.
                    print "beta_hat", beta_hat.get_shape()
                else:
                    beta_hat = beta
                    print "beta_hat", beta_hat.get_shape()

                # beta_hat = tf.expand_dims(beta_hat,1)
                # Distance of each data point from z0
                dist = (z - z0) ** 2
                dist = tf.reduce_sum(dist, 1)
                r = tf.sqrt(dist)
                # r= tf.sqrt(np.sum(((z-self.z0)**2),1))
                # m_of_beta = self.softplus(self.beta) # m(x)= log(1 + exp(x)) where x= w'*u. Last equation in A.2 Radial Flows.
                # beta_hat = -self.alpha + m_of_beta # It's a scalar.

                # h_alpha_r = self.get_h(r, alpha)  # Argument of h(.) in equation 14. (1000000,)
                h_alpha_r = 1 / (alpha + r)

                print "beta_hat", beta_hat.get_shape()
                beta_h_alpha_r = beta_hat * h_alpha_r
                print "beta_h_alpha_r", beta_h_alpha_r.get_shape()
                # fz = z + beta_hat * tf.mul(tf.transpose(tf.expand_dims(h_alpha_r, 1)),
                #                                            (z-self.z0))
                print "h_alpha_r shape", h_alpha_r.get_shape()
                # print "h_alpha_r shape", tf.expand_dims(h_alpha_r,1).get_shape()
                print "z shape", z.get_shape()
                # z = z + beta_hat * tf.multiply((z-z0), h_alpha_r)
                print "Shape 2nd term", tf.multiply(tf.multiply((z - z0), h_alpha_r), beta_hat).get_shape()
                # z = z + tf.multiply(tf.multiply((z-z0), h_alpha_r), beta_hat)
                z = z + tf.multiply(tf.multiply((z - z0), tf.expand_dims(h_alpha_r, 1)),
                                    tf.expand_dims(beta_hat, 1))
                # print "z shape", z.get_shape()
                # Calculation of log det jacobian
                print "r shape", r.get_shape()
                print "alpha shape", alpha.get_shape()

                # h_derivative_alpha_r = self.get_derivative_h(r, alpha)
                h_derivative_alpha_r = - 1 / ((alpha + r) ** 2)
                beta_h_derivative_alpha_r = beta_hat * h_derivative_alpha_r
                print "h_derivative_alpha_r shape", h_derivative_alpha_r.get_shape()
                print "beta_h_derivative_alpha_r shape", beta_h_derivative_alpha_r.get_shape()

                logdet_jacobian = tf.multiply(((1 + beta_h_alpha_r) ** (Z - 1)),
                                              (
                                              1 + beta_h_derivative_alpha_r * r + beta_h_alpha_r))  # Equation 14 second line.
                print "logdet_jacobian shape", logdet_jacobian.get_shape()
                log_detjs.append(tf.expand_dims(logdet_jacobian, 1))
                logdet_jacobian = tf.concat(log_detjs[0:K + 1], axis=1)
                sum_logdet_jacobian = tf.reduce_mean(logdet_jacobian)

                print "sum log det shape", sum_logdet_jacobian.get_shape()
                print "z shape", z.get_shape()
        return z, sum_logdet_jacobian