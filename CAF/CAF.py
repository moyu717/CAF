#coding:utf-8
import numpy as np
import tensorflow as tf
import random
from math import exp
import sys
import math
import scipy
import scipy.io
import logging
from attention import attention
from collections import *
import pandas as pd
import random
import logging
import attention

log = logging.getLogger()
log.setLevel(logging.INFO)
logFormatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')

fileHandler = logging.FileHandler('ML_CUL.log')
fileHandler.setFormatter(logFormatter)
log.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
log.addHandler(consoleHandler)


class caf:
    def __init__(self, nFlows, num_users, num_items, num_factors, input_dim, input_dim_user,batch_size,
                 dims, n_z, lr=0.1, verbose=True,
                 invert_condition=True, beta=True):
        self.nFlows = nFlows
        self.n_z = n_z
        self.m_num_users = num_users
        self.m_num_items = num_items
        self.m_num_factors = num_factors
        self.input_dim_user = input_dim_user
        self.batch_size = batch_size
        keep_prob = tf.placeholder(tf.float32)
        self.u = tf.placeholder(tf.int32, [None])
        self.i = tf.placeholder(tf.int32, [None])
        self.j = tf.placeholder(tf.int32, [None])

        self.m_U = 0.1 * np.random.randn(self.m_num_users, self.m_num_factors)
        self.m_theta_user = 0.1 * np.random.randn(self.m_num_users, self.m_num_factors)
        self.m_V = 0.1 * np.random.randn(self.m_num_items, self.m_num_factors)
        self.m_theta = 0.1 * np.random.randn(self.m_num_items, self.m_num_factors)
        self.item_bias = np.zeros(self.m_num_items)

        self.u_emb = tf.nn.embedding_lookup(self.m_U, self.u)
        self.i_emb = tf.nn.embedding_lookup(self.m_V, self.i)
        self.j_emb = tf.nn.embedding_lookup(self.m_V, self.j)
        self.i_item_bias = tf.nn.embedding_lookup(self.item_bias, self.i)
        self.j_item_bias = tf.nn.embedding_lookup(self.item_bias, self.j)
        self.x_pos_neg = tf.reduce_sum(tf.multiply(self.u_emb, (self.i_emb - self.j_emb)), 1, keep_dims=True)
        self.x_pos_neg_32 = tf.cast(self.x_pos_neg, dtype=tf.float32)
        self.mf_auc = tf.reduce_mean(tf.to_float(self.x_pos_neg_32 > 0))
        self.bias_regularization = 1.0
        self.user_regularization = 0.0025
        self.positive_item_regularization = 0.0025
        self.negative_item_regularization = 0.00025
        l2_norm = tf.add_n([
            tf.reduce_sum(self.user_regularization * tf.multiply(self.u_emb, self.u_emb)),
            tf.reduce_sum(self.positive_item_regularization * tf.multiply(self.i_emb, self.i_emb)),
            tf.reduce_sum(self.negative_item_regularization * tf.multiply(self.j_emb, self.j_emb)),
            tf.reduce_sum(self.bias_regularization * self.i_item_bias**2),
            tf.reduce_sum(self.bias_regularization * self.j_item_bias ** 2)
        ])

        l2_norm_32 = tf.cast(l2_norm, dtype=tf.float32)
        self.pos_neg_loss = l2_norm_32 - tf.reduce_mean(tf.log(tf.sigmoid(self.x_pos_neg_32)))

        self.input_dim = input_dim
        self.dims = dims
        self.lr = lr
        self.verbose = verbose
        self.invert_condition = invert_condition
        self.weights = []
        self.reg_loss = 0

        self.x_user = tf.placeholder(tf.float32, [None, self.input_dim_user], name='x')
        self.x = tf.placeholder(tf.float32, [None, self.input_dim], name='x')  #
        self.v = tf.placeholder(tf.float32, [None, self.m_num_factors])  #
        self.v_user = tf.placeholder(tf.float32, [None, self.m_num_factors])

        self.z0, self.z_k, sum_log_detj = self.inference(self.x, self.input_dim)
        x_recon, x_recons_mean, x_recons_logvar = self.generation(self.z_k)
        global_step = tf.Variable(0, trainable=False)
        self.recons_loss1, self.kl_loss1, self.loss_op = self.elbo_loss(self.x, x_recon, beta, global_step,
                            z_mu=self.z_mean, z_var=self.z_var, z0=self.z0,
                            zk=self.z_k, sum_log_detj=sum_log_detj)
        self.v_loss = 10.0 / 1.0 * tf.reduce_mean(tf.reduce_sum(tf.square(self.v - self.z0), 1))

        self.z_user, self.z_k_user, sum_log_detj_user = self.inference_user(self.x_user)
        x_recon_user, x_recons_mean_user, x_recons_logvar_user = self.generation_user(self.z_k_user)
        global_step = tf.Variable(0, trainable=False)
        self.recons_loss_user, self.kl_loss1_user, self.loss_op_user\
            = self.elbo_loss(self.x_user, x_recon_user, beta, global_step,
                            z_mu=self.z_mean_user, z_var=self.z_var_user, z0=self.z_user,
                            zk=self.z_k_user, sum_log_detj=sum_log_detj_user)

        self.v_user_loss = 10.0 / 1.0 * tf.reduce_mean(
            tf.reduce_sum(tf.square(self.v_user - self.z_user), 1))

        self.loss = self.loss_op + self.v_loss + 2e-4 * self.reg_loss + \
                    self.loss_op_user + self.v_user_loss + self.pos_neg_loss
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.saver = tf.train.Saver(self.weights)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        init = tf.global_variables_initializer()
        self.sess = tf.Session(config=config)
        self.sess.run(init)

    def inference(self, x, input_dim):
        with tf.variable_scope("inference", reuse=tf.AUTO_REUSE):
            rec = {'W1': tf.get_variable("W1", [input_dim, self.dims[0]],
                                         initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                   'b1': tf.get_variable("b1", [self.dims[0]],
                                         initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                   'W2': tf.get_variable("W2", [self.dims[0], self.dims[1]],
                                         initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                   'b2': tf.get_variable("b2", [self.dims[1]],
                                         initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                   'W_z_mean': tf.get_variable("W_z_mean", [self.dims[1], self.n_z],
                                               initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                   'b_z_mean': tf.get_variable("b_z_mean", [self.n_z],
                                               initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                   'W_z_log_sigma': tf.get_variable("W_z_log_sigma", [self.dims[1], self.n_z],
                                                    initializer=tf.contrib.layers.xavier_initializer(),
                                                    dtype=tf.float32),
                   'b_z_log_sigma': tf.get_variable("b_z_log_sigma", [self.n_z],
                                                    initializer=tf.constant_initializer(0.0), dtype=tf.float32)}
            with tf.variable_scope("flow"):
                nf = {'w_us': tf.get_variable("w_us", [self.dims[1], self.nFlows*self.n_z],
                                             initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                       'b_us': tf.get_variable("b_us", [self.nFlows*self.n_z],
                                             initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                       'w_ws': tf.get_variable("w_ws", [self.dims[1], self.nFlows*self.n_z],
                                             initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                       'b_ws': tf.get_variable("b_ws", [self.nFlows*self.n_z],
                                             initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                       'w_bs': tf.get_variable("w_bs", [self.dims[1], self.n_z],
                                                   initializer=tf.contrib.layers.xavier_initializer(),
                                                   dtype=tf.float32),
                       'b_bs': tf.get_variable("b_z_mean", [self.n_z],
                                                   initializer=tf.constant_initializer(0.0), dtype=tf.float32)}
        self.weights += [rec['W1'], rec['b1'], rec['W2'], rec['b2'], rec['W_z_mean'],
                         rec['b_z_mean'], rec['W_z_log_sigma'], rec['b_z_log_sigma']]
        self.reg_loss += tf.nn.l2_loss(rec['W1']) + tf.nn.l2_loss(rec['W2'])
        h1 = tf.nn.sigmoid(
            tf.matmul(x, rec['W1']) + rec['b1'])
        h2 = tf.nn.sigmoid(
            tf.matmul(h1, rec['W2']) + rec['b2'])
        h2 = tf.nn.dropout( h2, keep_prob)

        self.z_mean = tf.matmul(h2, rec['W_z_mean']) + rec['b_z_mean']  # 
        self.z_log_sigma_sq = tf.matmul(h2, rec['W_z_log_sigma']) + rec['b_z_log_sigma']  # 
        self.z_var = tf.exp(self.z_log_sigma_sq)
        us = tf.matmul(h2, nf['w_us']) + nf['b_us']
        ws = tf.matmul(h2, nf['w_ws']) + nf['b_ws']
        bs = tf.matmul(h2, nf['w_bs']) + nf['b_bs']
        self.flow_params = (us, ws, bs)  # 
        eps = tf.random_normal(shape=tf.shape(self.z_mean), mean=0, stddev=1)
        z0 = tf.add(self.z_mean, tf.multiply(tf.sqrt(self.z_var), eps))
        z_k, sum_log_detj = self.flow(z0, self.flow_params, self.nFlows,
                                                       self.n_z, self.invert_condition)
        return z0, z_k, sum_log_detj

    def inference_user(self, x):
        with tf.variable_scope("inference_user", reuse=tf.AUTO_REUSE):
            rec_user = {'W1': tf.get_variable("W1", [self.input_dim_user, self.dims[0]],
                                         initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                   'b1': tf.get_variable("b1", [self.dims[0]],
                                         initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                   'W2': tf.get_variable("W2", [self.dims[0], self.dims[1]],
                                         initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                   'b2': tf.get_variable("b2", [self.dims[1]],
                                         initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                   'W_z_mean': tf.get_variable("W_z_mean", [self.dims[1], self.n_z],
                                               initializer=tf.contrib.layers.xavier_initializer(),
                                               dtype=tf.float32),
                   'b_z_mean': tf.get_variable("b_z_mean", [self.n_z],
                                               initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                   'W_z_log_sigma': tf.get_variable("W_z_log_sigma", [self.dims[1], self.n_z],
                                                    initializer=tf.contrib.layers.xavier_initializer(),
                                                    dtype=tf.float32),
                   'b_z_log_sigma': tf.get_variable("b_z_log_sigma", [self.n_z],
                                                    initializer=tf.constant_initializer(0.0), dtype=tf.float32)}
            with tf.variable_scope("flow_user"):
                nf = {'w_us': tf.get_variable("w_us", [self.dims[1], self.nFlows * self.n_z],
                                              initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                      'b_us': tf.get_variable("b_us", [self.nFlows * self.n_z],
                                              initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                      'w_ws': tf.get_variable("w_ws", [self.dims[1], self.nFlows * self.n_z],
                                              initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                      'b_ws': tf.get_variable("b_ws", [self.nFlows * self.n_z],
                                              initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                      'w_bs': tf.get_variable("w_bs", [self.dims[1], self.n_z],
                                              initializer=tf.contrib.layers.xavier_initializer(),
                                              dtype=tf.float32),
                      'b_bs': tf.get_variable("b_z_mean", [self.n_z],
                                              initializer=tf.constant_initializer(0.0), dtype=tf.float32)}
        self.weights += [rec_user['W1'], rec_user['b1'], rec_user['W2'], rec_user['b2'], rec_user['W_z_mean'],
                         rec_user['b_z_mean'], rec_user['W_z_log_sigma'], rec_user['b_z_log_sigma']]
        self.reg_loss += tf.nn.l2_loss(rec_user['W1']) + tf.nn.l2_loss(rec_user['W2'])
        h1 = tf.nn.sigmoid(
            tf.matmul(x, rec_user['W1']) + rec_user['b1'])
        h2 = tf.nn.sigmoid(
            tf.matmul(h1, rec_user['W2']) + rec_user['b2'])
        h2 = tf.nn.dropout( h2, keep_prob)
        self.z_mean_user = tf.matmul(h2, rec_user['W_z_mean']) + rec_user['b_z_mean']
        self.z_log_sigma_sq_user = tf.matmul(h2, rec_user['W_z_log_sigma']) + rec_user[
            'b_z_log_sigma']
        self.z_var_user = tf.exp(self.z_log_sigma_sq)
        us_user = tf.matmul(h2, nf['w_us']) + nf['b_us']
        ws_user = tf.matmul(h2, nf['w_ws']) + nf['b_ws']
        bs_user = tf.matmul(h2, nf['w_bs']) + nf['b_bs']
        self.flow_params_user = (us_user, ws_user, bs_user)
        eps = tf.random_normal(shape=tf.shape(self.z_mean_user), mean=0, stddev=1)
        z0 = tf.add(self.z_mean_user, tf.multiply(tf.sqrt(self.z_var_user), eps))
        z_k, sum_log_detj = self.flow(z0, self.flow_params_user, self.nFlows,
                                                   self.n_z, self.invert_condition)
        return z0, z_k, sum_log_detj

    def generation(self, z_k):
        with tf.variable_scope("generation",reuse=tf.AUTO_REUSE):
            gen = {'w_h': tf.get_variable("w_h", [self.n_z, self.dims[1]],
                                         initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                   'b_h': tf.get_variable("b_h", [self.dims[1]],
                                         initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                   'w_mu': tf.get_variable("w_mu", [self.dims[1], self.dims[0]],
                                         initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                   'b_mu': tf.get_variable("b_mu", [self.dims[0]],
                                         initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                   'w_v': tf.get_variable("w_v", [self.dims[0], self.input_dim],
                                         initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                   'b_v': tf.get_variable("b_v", [self.input_dim],
                                         initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                   'w_v1': tf.get_variable("w_v1", [self.dims[0], self.input_dim],
                                          initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                   'b_v1': tf.get_variable("b_v1", [self.input_dim],
                                          initializer=tf.constant_initializer(0.0), dtype=tf.float32)}
        h2 = tf.nn.sigmoid(
            tf.matmul(z_k, gen['w_h']) + gen['b_h'])
        h1 = tf.nn.relu(
            tf.matmul(h2, gen['w_mu']) + gen['b_mu'])
        out_mu = tf.matmul(h1, gen['w_v']) + gen['b_v']
        
        out_log_var = tf.matmul(h1, gen['w_v1']) + gen['b_v1']
        out = tf.nn.sigmoid(out_mu)
        return out, out_mu, out_log_var

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
        h2 = tf.nn.sigmoid(
            tf.matmul(z_k, gen_user['w_h']) + gen_user['b_h'])
        h1 = tf.nn.relu(
            tf.matmul(h2, gen_user['w_mu']) + gen_user['b_mu'])
        out_mu = tf.matmul(h1, gen_user['w_v']) + gen_user['b_v']

        out_log_var = tf.matmul(h1, gen_user['w_v1']) + gen_user['b_v1']
        out = tf.nn.sigmoid(out_mu)
        return out, out_mu, out_log_var  # 

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
        mu = z_mu
        _var = z_var

        z0 = z0
        zk = zk
        logdet_jacobian = sum_log_detj

        # First term
        log_q0_z0 = self.gaussian_log_pdf(z0, mu, _var)
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

        recons_loss = log_p_x_given_zk
        kl_loss = log_qk_zk - log_p_zk

        recons_loss1 = tf.reduce_mean(log_p_x_given_zk)
        kl_loss1 = tf.reduce_mean(log_qk_zk - log_p_zk)
        print 'recons_loss1: ', recons_loss1
        print 'kl_loss1:', kl_loss1
        _elbo_loss = tf.reduce_mean(kl_loss + recons_loss)
        return recons_loss1, kl_loss1, _elbo_loss

    def cross_entropy_loss(self, prediction, actual, offset=1e-4):
        with tf.name_scope("cross_entropy"):
            _prediction = tf.clip_by_value(prediction, offset, 1 - offset)
            ce_loss = - tf.reduce_sum(actual * tf.log(_prediction)
                                      + (1 - actual) * tf.log(1 - _prediction), 1)
            return ce_loss

    def flow(self, z, flow_params, nFlows, z_dim,
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
                print "b shape", b.get_shape()
                if invert_condition:
                    uw = tf.reduce_sum(tf.matmul(w, u, transpose_a=True),
                                       axis=1, keep_dims=True)  
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
                psi = tf.reshape((1 - tf.tanh(zwb) ** 2), [-1, 1]) * w  
                # psi= tf.reduce_sum(tf.matmul(tf.transpose(1-self.tanh(zwb)**2), self.w))
                psi_u = tf.reduce_sum(tf.matmul(u_hat, psi, transpose_b=True),
                                      axis=1, keep_dims=True)
                logdet_jacobian = tf.log(tf.clip_by_value(tf.abs(1 + psi_u), 1e-4, 1e7))  
                log_detjs.append(logdet_jacobian)
                logdet_jacobian = tf.concat(log_detjs[0:nFlows + 1], axis=1)
                sum_logdet_jacobian = tf.reduce_mean(logdet_jacobian)
        return z, sum_logdet_jacobian

    def transform(self, data_x):
        data_en = self.sess.run(self.z_mean, feed_dict={self.x: data_x})
        return data_en
    def transform_user(self, data_x):
        data_en = self.sess.run(self.z_mean_user, feed_dict={self.x_user: data_x})
        return data_en
    def pmf_estimate(self, users, items):
        min_iter = 1
        max_iter = 10
        a_minus_b = 0.99  # 1 - 0.01
        converge = 1.0
        likelihood_old = 0.0
        likelihood = -math.exp(20)
        it = 0
        while ((it < max_iter and converge > 1e-6) or it < min_iter):
            likelihood_old = likelihood
            likelihood = 0
            ids = np.array([len(x) for x in items]) > 0
            v = self.m_V[ids]
            VVT = np.dot(v.T, v)
            XX = VVT * 1.0 + np.eye(self.m_num_factors) * 0.1
            for i in xrange(self.m_num_users):
                item_ids = users[i]
                n = len(item_ids)
                if n > 0:
                    A = np.copy(XX)
                    A += np.dot(self.m_V[item_ids, :].T, self.m_V[item_ids, :]) * a_minus_b  # a_minus_b = 0.99
                    B = np.copy(A)
                    A += np.eye(self.m_num_factors) * 0.1
                    x = 1.0 * np.sum(self.m_V[item_ids, :], axis=0) + 0.1 * self.m_theta_user[i, :]
                    self.m_U[i, :] = scipy.linalg.solve(A, x)
                    likelihood += -0.5 * n * 1.0
                    likelihood += 1.0 * np.sum(np.dot(self.m_V[item_ids, :], self.m_U[i, :][:, np.newaxis]),
                                                    axis=0)
                    likelihood += -0.5 * self.m_U[i, :].dot(B).dot(self.m_U[i, :][:, np.newaxis])
                    ep_user = self.m_U[i, :] - self.m_theta_user[i, :]
                    likelihood += -0.5 * 10.0 * np.sum(ep_user * ep_user)
                else:
                    A = np.copy(XX)
                    A += np.eye(self.m_num_factors) * 10.0
                    x = 10.0 * self.m_theta_user[i, :]
                    self.m_U[i, :] = scipy.linalg.solve(A, x)

                    ep_user = self.m_U[i, :] - self.m_theta_user[i, :]
                    likelihood += -0.5 * 10.0 * np.sum(ep_user * ep_user)
            ids = np.array([len(x) for x in users]) > 0
            u = self.m_U[ids]
            XX = np.dot(u.T, u) * 0.01
            for j in xrange(self.m_num_items):
                user_ids = items[j]
                m = len(user_ids)
                if m > 0:
                    A = np.copy(XX)  # 
                    A += np.dot(self.m_U[user_ids, :].T, self.m_U[user_ids, :]) * a_minus_b
                    B = np.copy(A)  # 0.52194386
                    A += np.eye(self.m_num_factors) * 10.0  # 10.0 = 10
                    x = 1.0 * np.sum(self.m_U[user_ids, :], axis=0) + 10.0 * self.m_theta[j, :]
                    self.m_V[j, :] = scipy.linalg.solve(A, x)  # 1.0 = 1

                    likelihood += -0.5 * m * 1.0  # 
                    likelihood += 1.0 * np.sum(np.dot(self.m_U[user_ids, :], self.m_V[j, :][:, np.newaxis]),
                                                    axis=0)
                    likelihood += -0.5 * self.m_V[j, :].dot(B).dot(self.m_V[j, :][:, np.newaxis])

                    ep = self.m_V[j, :] - self.m_theta[j, :]
                    likelihood += -0.5 * 10.0 * np.sum(ep * ep)
                else:
                    # m=0, this article has never been rated
                    A = np.copy(XX)
                    A += np.eye(self.m_num_factors) * 10.0
                    x = 10.0 * self.m_theta[j, :]
                    self.m_V[j, :] = scipy.linalg.solve(A, x)

                    ep = self.m_V[j, :] - self.m_theta[j, :]
                    likelihood += -0.5 * 10.0 * np.sum(ep * ep)
            it += 1
            converge = abs(
                1.0 * (likelihood - likelihood_old) / likelihood_old)  #

            if self.verbose:
                if likelihood < likelihood_old:
                    print("likelihood is decreasing!")

                print("[iter=%04d], likelihood=%.5f, converge=%.10f" % (it, likelihood, converge))

        return likelihood
    def cdl_estimate(self, data_x, data_x_user, num_iter, user_ratings, user_ratings_test, item_count, epoch):
        loss_app = []
        for i in range(num_iter):
            uij = self.generate_train_batch(user_ratings, user_ratings_test, item_count)
            b_x, ids = self.get_batch(data_x, self.batch_size)
            b_x_user, ids_user = self.get_batch(data_x_user, self.batch_size)
            _, all_loss, flow_loss, flow_user_loss, v_loss, recons_loss1, kl_loss1, z_k, z_k_user, z0, z0_user = \
                                    self.sess.run((self.optimizer, self.loss, self.loss_op, self.loss_op_user,
                                                   self.v_loss,
                                                   self.recons_loss1, self.kl_loss1, self.z_k,
                                                   self.z_k_user, self.z0, self.z_user),
                                                                    feed_dict={self.x: b_x, self.v: self.m_V[ids, :],
                                                                               self.x_user: b_x_user,
                                                                               self.v_user: self.m_U[ids_user, :],
                                                                               keep_prob: 0.5,
                                                              self.u: uij[:, 0], self.i: uij[:, 1], self.j: uij[:, 2]})

            loss_app.append(all_loss)
        print 'epoch ', epoch, ': ', np.mean(loss_app)
        return flow_loss, flow_user_loss

    def run(self, users, items, data_x, data_x_user, user_ratings, user_ratings_test):
        self.m_theta[:] = self.transform(data_x)
        self.m_V[:] = self.m_theta

        self.m_theta_user[:] = self.transform_user(data_x_user)
        self.m_U[:] = self.m_theta_user
        n = data_x.shape[0]

        n_user = data_x_user.shape[0]
        for epoch in range(100):
            num_iter = int(n / self.batch_size)
            flow_loss, flow_user_loss = self.cdl_estimate(data_x, data_x_user, num_iter, user_ratings, user_ratings_test, n, epoch)
            self.m_theta[:] = self.transform(data_x)
            likelihood = self.pmf_estimate(users, items)
            loss = -likelihood + 0.25 * flow_loss * n * 1.0 + 0.25 * flow_user_loss * n_user * 1.0
            logging.info("momo[#epoch=%06d], loss=%.5f, neg_likelihood=%.5f, flow_loss=%.5f" % (
                epoch, loss, -likelihood, flow_loss))
    def save_model(self, weight_path, pmf_path=None):
        self.saver.save(self.sess, weight_path)
        logging.info("Weights saved at " + weight_path)
        if pmf_path is not None:
            scipy.io.savemat(pmf_path, {"m_U": self.m_U, "m_V": self.m_V, "m_theta": self.m_theta})
            logging.info("Weights saved at " + pmf_path)

    def get_batch(self, X, size):
        ids = np.random.choice(len(X), size, replace=False)
        return (X[ids], ids)
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

    def noise_validator(self, noise, allowed_noises):
        try:
            if noise in allowed_noises:
                return True
            elif noise.split('-')[0] == 'mask' and float(noise.split('-')[1]):
                t = float(noise.split('-')[1])
                if t >= 0.0 and t <= 1.0:
                    return True
                else:
                    return False
        except:
            return False
        pass


# process data
def data_processing():
    data_path = '../ML/'
    file_train = open(data_path + 'ML_train.txt', 'r')
    f_1 = file_train.readlines()
    training_matrix_user = np.zeros((user_num, poi_num))
    training_matrix_user_1 = np.zeros((user_num, poi_num))
    training_matrix = np.zeros((poi_num, user_num))
    training_matrix_1 = np.zeros((poi_num, user_num))
    user_ratings = defaultdict(set)
    for lines in f_1:
        v = lines.strip().split()
        uid, iid, freq = int(v[0]), int(v[1]), int(float(v[2]))
        training_matrix[iid][uid] = freq
        training_matrix_user[uid][iid] = freq
        user_ratings[uid].add(iid)
    user_ratings_test = dict()
    for u, i_list in user_ratings.items():
        user_ratings_test[u] = random.sample(user_ratings[u], 1)[0]
    a = training_matrix.sum(axis=1)
    a_user = training_matrix_user.sum(axis=1)
    for lines in f_1:
        v = lines.strip().split()
        uid, iid, freq = int(v[0]), int(v[1]), int(float(v[2]))
        training_matrix_1[iid][uid] = float(freq / a[iid])
        training_matrix_user_1[uid][iid] = float(freq / a_user[uid])
    train_users = txt2array(data_path + "ML_train_user_item.txt")
    train_items = txt2array(data_path + "ML_train_item_user.txt")
    return train_users, train_items, training_matrix_1, training_matrix_user_1, user_ratings, user_ratings_test
def txt2array(file_path):
    arr = []
    for line in open(file_path):
        a = line.strip().split(',\t')
        if len(a[0:])==1:
            l = []
        else:
            try:
                l = [int(x) for x in a[1:]]
            except:
                print 1
        arr.append(l)
    return arr

train_users, train_items, training_matrix_1, training_matrix_user_1,\
user_ratings, user_ratings_test = data_processing()
latent_size = 300
user_num, item_num = 1000, 5000
CAF = caf(nFlows=32, num_users=user_num, num_items=item_num, num_factors=latent_size, batch_size=32,
    input_dim=user_num, input_dim_user=item_num,  dims=[600, 300], n_z=latent_size, lr=1e-6,
          verbose=False, invert_condition=True, beta=False)

CAF.run(train_users, train_items, training_matrix_1, training_matrix_user_1,
        user_ratings, user_ratings_test)
CAF.save_model(
    weight_path="model/caf_weight",
    pmf_path="model/caf_pmf")
