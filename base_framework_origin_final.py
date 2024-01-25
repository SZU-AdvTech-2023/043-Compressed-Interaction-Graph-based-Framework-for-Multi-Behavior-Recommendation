import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
import numpy as np
from Params import args
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
import Utils.NNLayers as NNs
from Utils.NNLayers import FC, Regularize, Activate, Dropout, Bias, getParam, defineParam, defineRandomNameParam
from DataHandler import negSamp, transpose, DataHandler, transToLsts
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
import pickle
import scipy.sparse as sp
from print_hook import PrintHook
import datetime
from time import time

class Recommender:
    def __init__(self, sess, handler):
        self.sess = sess
        self.handler = handler

        print('USER', args.user, 'ITEM', args.item)
        self.metrics = dict()
        self.weights = self._init_weights()      # 返回all_weights
        self.behEmbeds = NNs.defineParam('behEmbeds', [args.behNum, args.latdim // 2])    #   行为embeding  形状是[args.behNum, args.latdim // 2]
        if args.data == 'beibei':
            mets = ['Loss', 'preLoss', 'HR', 'NDCG', 'HR45', 'NDCG45', 'HR50', 'NDCG50', 'HR55', 'NDCG55', 'HR60', 'NDCG60', 'HR65', 'NDCG65', 'HR100', 'NDCG100']
        else:
            mets = ['Loss', 'preLoss', 'HR', 'NDCG', 'HR20', 'NDCG20', 'HR25', 'NDCG25', 'HR30', 'NDCG30', 'HR35', 'NDCG35', 'HR100', 'NDCG100']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()

    def makePrint(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '  '
        return ret

    def run(self):
        self.prepareModel()
        log('Model Prepared')
        if args.load_model != None:
            self.loadModel()
            stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
        else:
            stloc = 0
            init = tf.global_variables_initializer()
            self.sess.run(init)
            log('Variables Inited')
        train_time = 0
        test_time = 0
        for ep in range(stloc, args.epoch):
            test = (ep % args.tstEpoch == 0)
            t0 = time()
            reses = self.trainEpoch()
            t1 = time()
            train_time += t1-t0
            print('Train_time',t1-t0,'Total_time',train_time)
            log(self.makePrint('Train', ep, reses, test))
            if test:                  
                t2 = time()
                reses = self.testEpoch()
                t3 = time()
                test_time += t3-t2
                print('Test_time',t3-t2,'Total_time',test_time)
                log(self.makePrint('Test', ep, reses, test))                                                                    
            if ep % args.tstEpoch == 0:
                self.saveHistory()
            print()
        reses = self.testEpoch()
        log(self.makePrint('Test', args.epoch, reses, True))
        self.saveHistory()


    def messagePropagate(self, lats, adj):
        return Activate(tf.sparse_tensor_dense_matmul(adj, lats), self.actFunc)


    def defineModel(self):
        uEmbed0 = NNs.defineParam('uEmbed0', [args.user, args.latdim // 2], reg=True)   # 返回一个embeding(shape = user.num * 16)，tf张量
        iEmbed0 = NNs.defineParam('iEmbed0', [args.item, args.latdim // 2], reg=True)
        allEmbed = tf.concat([uEmbed0, iEmbed0], axis = 0)               # 比如两个矩阵分别是（2，3）和（3，3），concat后是（5，3），这里得到第0层的embeding

        self.ulat = [0] * (args.behNum)             # 结果是[0，0，0]
        self.ilat = [0] * (args.behNum)
        for beh in range(args.behNum):      # 行为循环
            ego_embeddings = allEmbed          
            all_embeddings = [ego_embeddings]           # 一个列表，来保存每一层的embedding,每一个都是tf.variable,也就是tensor张量
            if args.multi_graph == False:
                for index in range(args.gnn_layer):
                    symm_embeddings = tf.sparse_tensor_dense_matmul(self.adjs[beh], all_embeddings[-1])   
                    if args.encoder == 'lightgcn':
                        lightgcn_embeddings = symm_embeddings
                        all_embeddings.append(lightgcn_embeddings)
                    elif args.encoder == 'gccf':
                        gccf_embeddings = Activate(symm_embeddings, self.actFunc)
                        all_embeddings.append(gccf_embeddings)
                    elif args.encoder == 'gcn':
                        gcn_embeddings = Activate(
                            tf.matmul(symm_embeddings, self.weights['W_gc_%d' % index]) + self.weights[
                                'b_gc_%d' % index], self.actFunc)
                        all_embeddings.append(gcn_embeddings)
                    elif args.encoder == 'ngcf':
                        gcn_embeddings = Activate(
                            tf.matmul(symm_embeddings, self.weights['W_gc_%d' % index]) + self.weights[
                                'b_gc_%d' % index], self.actFunc)
                        bi_embeddings = tf.multiply(ego_embeddings, gcn_embeddings)
                        bi_embeddings = Activate(
                            tf.matmul(bi_embeddings, self.weights['W_bi_%d' % index]) + self.weights['b_bi_%d' % index],
                            self.actFunc)
                        all_embeddings.append(gcn_embeddings + bi_embeddings)

            elif args.multi_graph == True:     # 是-使用多图
                for index in range(args.gnn_layer):
                    if index == 0:
                        symm_embeddings = tf.sparse_tensor_dense_matmul(self.adjs[beh], all_embeddings[-1]) #这就相当于LightGCN中的E（k+1）= D-1/2AD-1/2E(k),得到K+1层的embedding
                        if args.encoder == 'lightgcn':              # 采用的就是LightGCN
                            lightgcn_embeddings = symm_embeddings      
                            all_embeddings.append(lightgcn_embeddings + all_embeddings[-1])     # 残差操作
                        elif args.encoder == 'gccf':
                            gccf_embeddings = Activate(symm_embeddings, self.actFunc)
                            all_embeddings.append(gccf_embeddings + all_embeddings[-1])
                        elif args.encoder == 'gcn':
                            gcn_embeddings = Activate(tf.matmul(symm_embeddings, self.weights['W_gc_%d' % index]) + self.weights['b_gc_%d' % index], self.actFunc)
                            all_embeddings.append(gcn_embeddings + all_embeddings[-1])
                        elif args.encoder == 'ngcf':
                            gcn_embeddings = Activate(tf.matmul(symm_embeddings, self.weights['W_gc_%d' % index]) + self.weights['b_gc_%d' % index], self.actFunc)
                            bi_embeddings = tf.multiply(ego_embeddings, gcn_embeddings)
                            bi_embeddings = Activate(tf.matmul(bi_embeddings, self.weights['W_bi_%d' % index]) + self.weights['b_bi_%d' % index], self.actFunc)
                            all_embeddings.append(gcn_embeddings + bi_embeddings + all_embeddings[-1])
                    else:       # 就是L=2的时候
                        atten = FC(ego_embeddings, args.behNum, reg=True, useBias=True,
                                   activation=self.actFunc, name='attention_%d_%d'%(beh,index), reuse=True) # reuse=True，如果出现同名的FC，那么将不会再被创建，而是使用之前的。   这里的beh和index相当于公式中的k和l    shape是(29693, 3)
                        temp_embeddings = []
                        for inner_beh in range(args.behNum):    
                            neighbor_embeddings = tf.sparse_tensor_dense_matmul(self.adjs[inner_beh], symm_embeddings)  # 第二层卷积
                            temp_embeddings.append(neighbor_embeddings)      # 生成3个卷积后的，形状是(29693, 16)
                        all_temp_embeddings = tf.stack(temp_embeddings, 1)             #shape是(29693, 3，16)
                        
                        #如果我们有两个形状为 (3, 4) 的张量，使用 tf.stack 在 axis=0 上堆叠，那么新张量的形状将是 (2, 3, 4)。如果在 axis=1 上堆叠，新张量的形状将是 (3, 2, 4)。
                        symm_embeddings = tf.reduce_sum(tf.einsum('abc,ab->abc', all_temp_embeddings, atten), axis=1, keepdims=False)  
                        # 操作6，symm_embeddings得到的是一个shape为（29963.16）
                        if args.encoder == 'lightgcn':
                            lightgcn_embeddings = symm_embeddings
                            all_embeddings.append(lightgcn_embeddings + all_embeddings[-1])
                        elif args.encoder == 'gccf':
                            gccf_embeddings = Activate(symm_embeddings, self.actFunc)
                            all_embeddings.append(gccf_embeddings + all_embeddings[-1])
                        elif args.encoder == 'gcn':
                            gcn_embeddings = Activate(tf.matmul(symm_embeddings, self.weights['W_gc_%d' % index]) + self.weights['b_gc_%d' % index], self.actFunc)
                            all_embeddings.append(gcn_embeddings + all_embeddings[-1])
                        elif args.encoder == 'ngcf':
                            gcn_embeddings = Activate(tf.matmul(symm_embeddings, self.weights['W_gc_%d' % index]) + self.weights['b_gc_%d' % index], self.actFunc)
                            bi_embeddings = tf.multiply(ego_embeddings, gcn_embeddings)
                            bi_embeddings = Activate(tf.matmul(bi_embeddings, self.weights['W_bi_%d' % index]) + self.weights['b_bi_%d' % index], self.actFunc)
                            all_embeddings.append(gcn_embeddings + bi_embeddings + all_embeddings[-1])

            all_embeddings = tf.add_n(all_embeddings)       # 得到beh的embeding  # shape(user.number+item.number,16)
            self.ulat[beh], self.ilat[beh] = tf.split(all_embeddings, [args.user, args.item], 0)    # 得到用户和物品的embeding,列表中的数据是tensorflow张量
        self.ulat_merge, self.ilat_merge = tf.add_n(self.ulat), tf.add_n(self.ilat)


    def _init_weights(self):
        all_weights = dict()
        initializer = tf.random_normal_initializer(stddev=0.01)  

        self.weight_size_list = [args.latdim // 2] + [args.latdim // 2] * args.gnn_layer    # 得到[16,16,16]  3层卷积

        for k in range(args.gnn_layer):
            all_weights['W_gc_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_gc_%d' % k)
            all_weights['b_gc_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_gc_%d' % k)

            all_weights['W_bi_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_%d' % k)
            all_weights['b_bi_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_bi_%d' % k)

            all_weights['W_mlp_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_mlp_%d' % k)
            all_weights['b_mlp_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_mlp_%d' % k)

        return all_weights

    def bilinear_predict(self, src):
        uids = self.uids[src]
        iids = self.iids[src]

        src_ulat = tf.nn.embedding_lookup(self.ulat[src], uids)
        src_ilat = tf.nn.embedding_lookup(self.ilat[src], iids)

        behEmbed = self.behEmbeds[src]
        predEmbed = tf.reduce_sum(src_ulat * src_ilat * tf.expand_dims(behEmbed, axis=0), axis=-1, keep_dims=False)
        preds = predEmbed

        return preds * args.mult

    def shared_bottom_predict(self, src):
        uids = self.uids[src]
        iids = self.iids[src]

        src_ulat = tf.nn.embedding_lookup(self.ulat_merge, uids)
        src_ilat = tf.nn.embedding_lookup(self.ilat_merge, iids)

        preds = tf.squeeze(FC(tf.concat([src_ulat,src_ilat], axis=-1), 1, reg=True, useBias=True,
                     name='tower_' + str(src), reuse=True))

        return preds * args.mult

    def mmoe_predict(self, src):
        uids = self.uids[src]
        iids = self.iids[src]

        src_ulat = tf.nn.embedding_lookup(self.ulat_merge, uids)
        src_ilat = tf.nn.embedding_lookup(self.ilat_merge, iids)

        exper_info = []
        for i in range(args.num_exps):
            exper_net = FC(tf.concat([src_ulat,src_ilat], axis=-1), args.latdim, reg=True, useBias=True,
                     activation=self.actFunc, name='expert_' + str(i), reuse=True)
            exper_info.append(exper_net)
        expert_concat = tf.stack(exper_info, axis = 1)

        gate_out = FC(tf.concat([src_ulat,src_ilat], axis=-1), args.num_exps, reg=True, useBias=True,
                    activation='softmax', name='gate_softmax_' + str(src), reuse=True)
        
        mmoe_out = tf.reduce_sum(tf.expand_dims(gate_out, axis = -1) * expert_concat, axis=1, keep_dims=False)

        preds = tf.squeeze(FC(mmoe_out, 1, reg=True, useBias=True,
                    name='tower_' + str(src), reuse=True))

        return preds * args.mult

    def ple_predict(self, src):
        uids = self.uids[src]
        iids = self.iids[src]

        src_ulat = tf.nn.embedding_lookup(self.ulat_merge, uids)
        src_ilat = tf.nn.embedding_lookup(self.ilat_merge, iids)

        def cgc_net(level_name):
            specific_expert_outputs = []
            if args.num_exps == 3:
                specific_expert_num = 1
                shared_expert_num = 1
            else:
                specific_expert_num = 2
                shared_expert_num = 1
            for i in range(specific_expert_num):
                expert_network = FC(tf.concat([src_ulat, src_ilat], axis=-1), args.latdim, reg=True, useBias=True,
                                    activation=self.actFunc, name=level_name + '_expert_specific_' + str(i) + str(src),
                                    reuse=True)
                specific_expert_outputs.append(expert_network)
            shared_expert_outputs = []
            for k in range(shared_expert_num):
                expert_network = FC(tf.concat([src_ulat, src_ilat], axis=-1), args.latdim, reg=True, useBias=True,
                                    activation=self.actFunc, name=level_name + 'expert_shared_' + str(k), reuse=True)
                shared_expert_outputs.append(expert_network)

            cur_expert_num = specific_expert_num + shared_expert_num
            cur_experts = specific_expert_outputs + shared_expert_outputs

            expert_concat = tf.stack(cur_experts, axis=1)

            gate_out = FC(tf.concat([src_ulat, src_ilat], axis=-1), cur_expert_num, reg=True, useBias=True,
                          activation='softmax', name='gate_softmax_' + str(src), reuse=True)
            gate_out = tf.expand_dims(gate_out, axis=-1)

            gate_mul_expert = tf.reduce_sum(expert_concat * gate_out, axis=1, keep_dims=False)
            return gate_mul_expert

        ple_outputs = cgc_net(level_name='level_')
        preds = tf.squeeze(FC(ple_outputs, 1, reg=True, useBias=True,
                    name='tower_' + str(src), reuse=True))
        return preds * args.mult

    def sesg_predict(self, src):
        uids = self.uids[src]
        iids = self.iids[src]

        src_ulat = tf.nn.embedding_lookup(self.ulat[src], uids)
        src_ilat = tf.nn.embedding_lookup(self.ilat[src], iids)

        metalat111 = FC(tf.concat([src_ulat, src_ilat], axis=-1), args.behNum, reg=True, useBias=True,
                        activation='softmax', name='gate111', reuse=True)
        w1 = tf.reshape(metalat111, [-1, args.behNum, 1])
        
        exper_info = []
        for index in range(args.behNum):
            exper_info.append(
                tf.nn.embedding_lookup(self.ulat[index], uids) * tf.nn.embedding_lookup(self.ilat[index], iids))
        predEmbed = tf.stack(exper_info, axis=2)
        sesg_out = tf.reshape(predEmbed @ w1, [-1, args.latdim // 2])

        preds = tf.squeeze(tf.reduce_sum(sesg_out, axis=-1))

        return preds * args.mult

    def sesgpro_predict(self, src):
        uids = self.uids[src]
        iids = self.iids[src]

        src_ulat = tf.nn.embedding_lookup(self.ulat[src], uids)
        src_ilat = tf.nn.embedding_lookup(self.ilat[src], iids)

        metalat111 = FC(tf.concat([src_ulat, src_ilat], axis=-1), args.behNum + 1, reg=True, useBias=True,
                        activation='softmax', name='gate111', reuse=True)
        w1 = tf.reshape(metalat111, [-1, args.behNum + 1, 1])
        
        exper_info = []
        for index in range(args.behNum + 1):
            if index < args.behNum:
                exper_info.append(
                    tf.nn.embedding_lookup(self.ulat[index], uids) * tf.nn.embedding_lookup(self.ilat[index], iids))
            if index == args.behNum:
                exper_info.append(
                    tf.nn.embedding_lookup(self.ulat_merge, uids) * tf.nn.embedding_lookup(self.ilat_merge, iids))
        predEmbed = tf.stack(exper_info, axis=2)
        sesg_out = tf.reshape(predEmbed @ w1, [-1, args.latdim // 2])

        preds = tf.squeeze(tf.reduce_sum(sesg_out, axis=-1))

        return preds * args.mult

    def create_multiple_adj_mat(self, adj_mat):      # 生成三种归一化矩阵，分别是左归一，右归一，和对称归一。
        def left_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            print('generate left_adj_single adjacency matrix.')
            return norm_adj.tocoo()

        def right_adj_single(adj):
            rowsum = np.array(adj.sum(0))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = adj.dot(d_mat_inv)
            print('generate right_adj_single adjacency matrix.')
            return norm_adj.tocoo()

        def symm_adj_single(adj_mat):
            rowsum = np.array(adj_mat.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            rowsum = np.array(adj_mat.sum(0))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv_trans = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv_trans)
            print('generate symm_adj_single adjacency matrix.')
            return norm_adj.tocoo()

        left_adj_mat = left_adj_single(adj_mat)
        right_adj_mat = right_adj_single(adj_mat)
        symm_adj_mat = symm_adj_single(adj_mat)

        return left_adj_mat.tocsr(), right_adj_mat.tocsr(), symm_adj_mat.tocsr()

    def mult(self,a,b):
        return (a*b).sum(1)
    
    def coef(self,buy_mat,view_mat):
        buy_dense = np.array(buy_mat.todense())
        view_dense = np.array(view_mat.todense())
        buy = buy_dense-buy_dense.sum(1).reshape(-1,1)/buy_dense.shape[1]
        view = view_dense-view_dense.sum(1).reshape(-1,1)/view_dense.shape[1]
        return self.mult(buy,view)/np.sqrt((self.mult(buy,buy))*self.mult(view,view))
      
    def mult_tmall(self,a,b):
        return a.multiply(b).sum(1)
    
    def coef_tmall(self,buy_mat,view_mat):
        buy = buy_mat
        view = view_mat
        return np.array(self.mult_tmall(buy,view))/np.sqrt(np.array(self.mult_tmall(buy,buy))*np.array(self.mult_tmall(view,view)))
    
    def prepareModel(self):
        self.actFunc = 'leakyRelu'
        self.adjs = []                # 得到self.adjs是一个数组，对应三个归一化完的矩阵，就是对应A1，A2，A3（三种行为），张量SparseTensor类型
        self.uids, self.iids = [], []   # 三种行为的用户和物品id
        self.left_trnMats, self.right_trnMats, self.symm_trnMats, self.none_trnMats = [], [], [], []

        for i in range(args.behNum):
            R = self.handler.trnMats[i].tolil()   # R.tolil() 将矩阵 R 转换为 List of Lists (LIL) 格式。LIL 格式以两个列表来存储矩阵：一个列表存储非零元素的值，另一个列表存储这些非零元素的位置。这种格式便于逐行修改矩阵。
            
            coomat = sp.coo_matrix(R)
            coomat_t = sp.coo_matrix(R.T)
            row = np.concatenate([coomat.row, coomat_t.row + R.shape[0]])
            col = np.concatenate([R.shape[0] + coomat.col, coomat_t.col])
            data = np.concatenate([coomat.data.astype(np.float32), coomat_t.data.astype(np.float32)])
            adj_mat = sp.coo_matrix((data, (row, col)), shape=(args.user + args.item, args.user + args.item))
            
            left_trn, right_trn, symm_trn = self.create_multiple_adj_mat(adj_mat)   # 生成三个归一化矩阵
            self.left_trnMats.append(left_trn)
            self.right_trnMats.append(right_trn)
            self.symm_trnMats.append(symm_trn)
            self.none_trnMats.append(adj_mat.tocsr())
        #  生成的归一化矩阵列表
        if args.normalization == "left":
            self.final_trnMats = self.left_trnMats
        elif args.normalization == "right":
            self.final_trnMats = self.right_trnMats
        elif args.normalization == "symm":
            self.final_trnMats = self.symm_trnMats
        elif args.normalization == 'none':
            self.final_trnMats = self.none_trnMats
        for i in range(args.behNum):     # 得到self.adjs，然后占位符elf.uids, self.iids
            adj = self.final_trnMats[i]
            # print("-------------------------------------")
            # print(type(adj))     # <class 'scipy.sparse.csr.csr_matrix'>
            # break
            idx, data, shape = transToLsts(adj, norm=False)# idx：一个二维数组，表示非零元素的位置。每个元素是 [row, col] 格式的列表，表示矩阵中一个非零元素的行和列索引。data：一个一维数组，包含与 idx 中对应位置的非零元素的值。shape：一个列表，表示原始矩阵的维度（行数和列数）。
            self.adjs.append(tf.sparse.SparseTensor(idx, data, shape))  # self.adjs是一个数组，对应三个归一化完的矩阵 主要目的就是把原来的矩阵，变成张量SparseTensor类型
            self.uids.append(tf.placeholder(name='uids' + str(i), dtype=tf.int32, shape=[None]))   # 占位符，可以接收任意长度的一维数组
            self.iids.append(tf.placeholder(name='iids' + str(i), dtype=tf.int32, shape=[None]))
        self.defineModel()
        self.preLoss = 0
        for src in range(args.behNum):
            if args.decoder == 'single':
                if src != args.behNum-1:
                    continue
                preds = self.shared_bottom_predict(src)
            elif args.decoder == 'bilinear':
                preds = self.bilinear_predict(src)
            elif args.decoder == 'shared_bottom':
                preds = self.shared_bottom_predict(src)
            elif args.decoder == 'mmoe':
                preds = self.mmoe_predict(src)
            elif args.decoder == 'ple':
                preds = self.ple_predict(src)
            elif args.decoder == 'sesg':
                preds = self.sesg_predict(src)
            elif args.decoder == 'sesgpro':
                preds = self.sesgpro_predict(src)

            sampNum = tf.shape(self.uids[src])[0] // 2
            posPred = tf.slice(preds, [0], [sampNum])
            negPred = tf.slice(preds, [sampNum], [-1])
            self.preLoss += tf.reduce_mean(tf.nn.softplus(-(posPred - negPred)))
            if src == args.behNum - 1:
                self.targetPreds = preds
        self.regLoss = args.reg * Regularize()
        
        
        self.loss = self.preLoss + self.regLoss

        globalStep = tf.Variable(0, trainable=False)
        learningRate = tf.train.exponential_decay(args.lr, globalStep, args.decay_step, args.decay, staircase=True)
        self.optimizer = tf.train.AdamOptimizer(learningRate).minimize(self.loss, global_step=globalStep)

    def sampleTrainBatch(self, batIds, labelMat):
        temLabel = labelMat[batIds].toarray()
        batch = len(batIds)
        temlen = batch * 2 * args.sampNum
        uLocs = [None] * temlen
        iLocs = [None] * temlen
        cur = 0
        for i in range(batch):
            posset = np.reshape(np.argwhere(temLabel[i] != 0), [-1])
            sampNum = min(args.sampNum, len(posset))
            if sampNum == 0:
                poslocs = [np.random.choice(args.item)]
                neglocs = [poslocs[0]]
            else:
                poslocs = np.random.choice(posset, sampNum)
                neglocs = negSamp(temLabel[i], sampNum, args.item)
            for j in range(sampNum):
                posloc = poslocs[j]
                negloc = neglocs[j]
                uLocs[cur] = uLocs[cur + temlen // 2] = batIds[i]
                iLocs[cur] = posloc
                iLocs[cur + temlen // 2] = negloc
                cur += 1
        uLocs = uLocs[:cur] + uLocs[temlen // 2: temlen // 2 + cur]
        iLocs = iLocs[:cur] + iLocs[temlen // 2: temlen // 2 + cur]
        return uLocs, iLocs

    def trainEpoch(self):
        num = args.user
        sfIds = np.random.permutation(num)[:args.trnNum]
        epochLoss, epochPreLoss = [0] * 2
        num = len(sfIds)
        steps = int(np.ceil(num / args.batch))
        for i in range(steps):
            st = i * args.batch
            ed = min((i + 1) * args.batch, num)
            batIds = sfIds[st: ed]

            target = [self.optimizer, self.preLoss, self.regLoss, self.loss]
            feed_dict = {}
            for beh in range(args.behNum):
                uLocs, iLocs = self.sampleTrainBatch(batIds, self.handler.trnMats[beh])
                feed_dict[self.uids[beh]] = uLocs
                feed_dict[self.iids[beh]] = iLocs

            res = self.sess.run(target, feed_dict=feed_dict,
                                options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
            preLoss, regLoss, loss = res[1:]

            epochLoss += loss
            epochPreLoss += preLoss             
        ret = dict()
        ret['Loss'] = epochLoss / steps
        ret['preLoss'] = epochPreLoss / steps
        return ret

    def sampleTestBatch(self, batIds, labelMat):
        batch = len(batIds)                              # batIds是nparray类型
        temTst = self.handler.tstInt[batIds]
        # temTst = np.array([self.handler.tstInt[bId][0] for bId in batIds])      # 在这行代码 temTst = np.array([self.handler.tstInt[bId] for bId in batIds]) 中，self.handler.tstInt 的每个键对应的值是一个列表（例如 [2756]），那么 temTst就会成为一个包含这些列表的数组。当您访问 temTst[i]时，得到的是一个列表而不是单个的整数。  所以这里相当于一个二维的list-----> 二维数组
        temLabel = labelMat[batIds].toarray()
        temlen = batch * 100
        uLocs = [None] * temlen
        iLocs = [None] * temlen
        tstLocs = [None] * batch
        cur = 0
        for i in range(batch):
            posloc = temTst[i]
            negset = np.reshape(np.argwhere(temLabel[i] == 0), [-1])
            rdnNegSet = np.random.permutation(negset)[:99]
            locset = np.concatenate((rdnNegSet, np.array([posloc])))     
            tstLocs[i] = locset
            for j in range(100):
                uLocs[cur] = batIds[i]
                iLocs[cur] = locset[j]
                cur += 1
        return uLocs, iLocs, temTst, tstLocs

# 采样要改成全采样 既99 ---> 全部用户
    def testEpoch(self):
        epochHit, epochNdcg = [0] * 2 
        
        ids = self.handler.tstUsrs
        num = len(ids)
        tstBat = args.batch
        steps = int(np.ceil(num / tstBat))
        for i in range(steps):
            st = i * tstBat
            ed = min((i + 1) * tstBat, num)
            batIds = ids[st: ed]
            feed_dict = {}
            uLocs, iLocs, temTst, tstLocs = self.sampleTestBatch(batIds, self.handler.trnMats[-1])
            feed_dict[self.uids[-1]] = uLocs
            feed_dict[self.iids[-1]] = iLocs
            preds = self.sess.run(self.targetPreds, feed_dict=feed_dict,
                                  options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
            hit, ndcg = self.calcRes(np.reshape(preds, [ed - st, 100]), temTst, tstLocs)
            epochHit += hit
            epochNdcg += ndcg
        
        ret = dict()
        ret['HR'] = epochHit / num
        ret['NDCG'] = epochNdcg / num
        return ret

    def calcRes(self, preds, temTst, tstLocs):
        hit = 0
        ndcg = 0
        for j in range(preds.shape[0]):
            predvals = list(zip(preds[j], tstLocs[j]))
            predvals.sort(key=lambda x: x[0], reverse=True)
            shoot = list(map(lambda x: x[1], predvals[:args.shoot]))
            if temTst[j] in shoot:
                hit += 1
                ndcg += np.reciprocal(np.log2(shoot.index(temTst[j]) + 2))        # ndcg的计算问题，因为采样改成了全采样
        return hit, ndcg

    def saveHistory(self):
        if args.epoch == 0:
            return
        with open('History/' + args.save_path + '.his', 'wb') as fs:
            pickle.dump(self.metrics, fs)

        saver = tf.train.Saver()
        saver.save(self.sess, 'Models/' + args.save_path)
        log('Model Saved: %s' % args.save_path)

    def loadModel(self):
        saver = tf.train.Saver()
        saver.restore(sess, 'Models/' + args.load_model)
        with open('History/' + args.load_model + '.his', 'rb') as fs:
            self.metrics = pickle.load(fs)
        log('Model Loaded')


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    log_dir = 'log/' + args.data + '/' + os.path.basename(__file__)

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    log_file = open(log_dir + '/log' + str(datetime.datetime.now()), 'w')

    def my_hook_out(text):
        log_file.write(text)
        log_file.flush()
        return 1, 0, text

    ph_out = PrintHook()
    ph_out.Start(my_hook_out)

    print("Use gpu id:", args.gpu_id)
    for arg in vars(args):
        print(arg + '=' + str(getattr(args, arg)))

    logger.saveDefault = True
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    log('Start')
    handler = DataHandler()
    handler.LoadData()
    log('Load Data')

    with tf.Session(config=config) as sess:
        recom = Recommender(sess, handler)
        recom.run()