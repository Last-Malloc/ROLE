import tensorflow as tf
from util.cnn import fc_layer as fc
import vs_multilayer
from dataset import TestingDataSet
from dataset import TrainingDataSet
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
wordembed_params = './word_embedding/embed_matrix.npy'
embedding_mat = np.load(wordembed_params)
class ROLE_Model(object):
    def __init__(self, batch_size, train_visual_feature_dir, test_visual_feature_dir, lr, n_input_visual, n_input_text,
                   n_hidden_text, n_step_text,train_sliding_dir,test_sliding_dir,semantic_size,mpl_hidden ):
        self.batch_size = batch_size
        self.test_batch_size = 1
        self.vs_lr = lr
        self.n_input_visual = n_input_visual # the size of visual and semantic comparison size
        self.n_input_text = n_input_text
        self.n_step_text = n_step_text
        self.n_hidden_text=n_hidden_text
        self.context_size = 128
        self.context_num = 1
        self.visual_feature_dim=4096
        self.lambda_regression = 0.01
        self.alpha = 1.0 / batch_size
        self.train_set=TrainingDataSet(train_visual_feature_dir, self.batch_size,train_sliding_dir)
        self.test_set=TestingDataSet(test_visual_feature_dir, self.test_batch_size,test_sliding_dir)
        self.semantic=semantic_size
        self.mpl_hidden=mpl_hidden
    '''
    used in training alignment model, ROLE(aln)
    '''
    def fill_feed_dict_train(self):
        image_batch,sentence_batch,offset_batch = self.train_set.next_batch()
        input_feed = {
                self.visual_featmap_ph_train: image_batch,
                self.sentence_ph_train: sentence_batch,
                self.offset_ph: offset_batch
        }

        return input_feed

    '''
    used in training alignment+regression model, ROLE(reg)
    '''
    def fill_feed_dict_train_reg(self):
        image_batch, sentence_batch, offset_batch = self.train_set.next_batch_iou()
        input_feed = {
                self.visual_featmap_ph_train: image_batch,
                self.sentence_ph_train: sentence_batch,
                self.offset_ph: offset_batch
        }

        return input_feed

    def bilstm(self,x):
        """RNN (LSTM or GRU) model for image"""
        # x.shape [N,T,D]
        x=tf.transpose(x,[1,0,2])# [T,N,D]
        fw_x = tf.reshape(x, [-1, self.n_input_text]) # step*batch, feature
        fw_x = tf.split(0, self.n_step_text, fw_x)
        with tf.variable_scope('bilstm_lt'):
            #one-layer bilstm
            lstm_fw_cell = rnn_cell.BasicLSTMCell(self.n_hidden_text, forget_bias=1.0, state_is_tuple=True)
            lstm_bw_cell = rnn_cell.BasicLSTMCell(self.n_hidden_text, forget_bias=1.0, state_is_tuple=True)
            #dropout
            #lstm_fw_cell = rnn_cell.DropoutWrapper(cell=lstm_fw_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
            #lstm_bw_cell = rnn_cell.DropoutWrapper(cell=lstm_bw_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
            with tf.variable_scope('fw_lt'):
                 (output_fw, state_fw) = rnn.rnn(lstm_fw_cell,fw_x,dtype=tf.float32)
                 t=tf.convert_to_tensor(output_fw)
                 print (t.get_shape().as_list())
            # backward direction
            with tf.variable_scope('bw_lt'):
                bw_x = tf.reverse(x, [True,False,False])# reverse time dim
                bw_x = tf.reshape(bw_x, [-1, self.n_input_text])  # step*batch, feature
                bw_x = tf.split(0, self.n_step_text, bw_x)
                (output_bw, state_bw) = rnn.rnn(lstm_bw_cell,bw_x,dtype=tf.float32)
            # output_bw.shape = [timestep_size, batch_size, hidden_size]
            output_bw = tf.reverse(output_bw, [True,False,False])
            output = tf.concat(2,[output_fw, output_bw])
        return output
    '''
    cross modal processing module
    '''
    def cross_modal_comb(self, input_vision_obj1,BoW_obj1,batch_size):
        vv_feature= tf.reshape(tf.tile(input_vision_obj1, [batch_size, 1]),[batch_size, batch_size, self.visual_feature_dim*(2*self.context_num+1)])
        ss_feature= tf.reshape(tf.tile(BoW_obj1,[1, batch_size]),[batch_size, batch_size, self.n_input_text])
        concat_feature = tf.reshape(tf.concat(2,[vv_feature, ss_feature]),[1,batch_size, batch_size, self.visual_feature_dim*(2*self.context_num+1)+self.n_input_text])
        return concat_feature

    '''
    visual semantic inference, including visual semantic alignment and clip location regression
    '''
    def visual_semantic_infer(self,visual_feature_train, text_feature_train, visual_feature_test, text_feature_test):
       name="CTRL_Model"
       with tf.variable_scope(name):

         # text_feature_train [N,T] shape word index matrix
         # 0. Word embedding
         # text_seq has shape [N, T] and embedded_seq has shape [N, T, D].
         embedded_seq_train = tf.nn.embedding_lookup(embedding_mat, text_feature_train)
         # 1. Encode the sentence into a vector representation, using the final
         # hidden states in a one-layer bidirectional LSTM network
         q_reshape=self.bilstm(embedded_seq_train)
         print (q_reshape.get_shape().as_list())
         # 2. attention units over the words in each sentence fc(fc(Q+C+PRE+POST))
         q_reshape_flat = tf.reshape(q_reshape, [self.n_step_text * self.batch_size, self.n_hidden_text * 2])
         visual_feature_train=tf.transpose(visual_feature_train, [0, 2, 1])  # batch ctx fea
         visual_train=tf.reshape(visual_feature_train,[self.batch_size*(2*self.context_num+1),self.visual_feature_dim])
         query_term=fc('q2s_lt', q_reshape_flat, output_dim=self.semantic)
         moment_term=fc('c2s_lt', visual_train, output_dim=self.semantic)

         query_term=tf.reshape(query_term,[self.batch_size,self.n_step_text,self.semantic])
         moment_term=tf.reshape( moment_term,[self.batch_size,2*self.context_num+1,self.semantic])
         term2=tf.reduce_sum(moment_term,1,keep_dims=True)
         term2=tf.reshape(term2,[self.batch_size,self.semantic])
         term2=tf.reshape(tf.tile(term2,[1,self.n_step_text]),[self.batch_size,self.n_step_text,self.semantic])
         term=tf.nn.relu(tf.reshape(tf.add(query_term,term2), [self.n_step_text * self.batch_size, self.semantic]))
         scores_obj1 = fc('fc_scores_query_lt',term, output_dim=1)
         scores_obj1_train=tf.reshape(scores_obj1,[self.batch_size,self.n_step_text])
         is_not_pad=tf.cast(tf.not_equal(text_feature_train,0),tf.float32)
         #probs_obj1=tf.nn.softmax(scores_obj1_train)
         #probs_obj1=tf.mul(probs_obj1,is_not_pad)
         probs_obj1 = tf.mul(scores_obj1_train,is_not_pad)
         probs_obj1 = probs_obj1 / tf.reduce_sum(probs_obj1, 1, keep_dims=True)

         temp1= tf.transpose(tf.reshape(tf.tile(probs_obj1, [1, self.n_input_text]),[self.batch_size, self.n_input_text, self.n_step_text]),[0,2,1])  # [N,T,embed_dim]
         BoW_obj1 = tf.reduce_sum(tf.mul(temp1,embedded_seq_train), reduction_indices=1)
         print (BoW_obj1.get_shape().as_list())
         #3.0 visual attention part: x_i=Wv_i+b then softmax(x_i.q_j) output: [N, visual_feature_size]

         input_vision_obj1=tf.reshape(visual_feature_train,[self.batch_size,-1])
         # cross-modal part
         transformed_clip_train_norm = tf.nn.l2_normalize(input_vision_obj1 , dim=1)
         transformed_obj1_sent_train_norm = tf.nn.l2_normalize(BoW_obj1, dim=1)
         cross_modal_vec_train = self.cross_modal_comb(transformed_clip_train_norm,transformed_obj1_sent_train_norm, self.batch_size)  # batch batch 2*conmmon_space_dim
         sim_score_mat_train = vs_multilayer.vs_multilayer(cross_modal_vec_train, "vs_multilayer_lt",self.mpl_hidden)
         sim_score_mat_train = tf.reshape(sim_score_mat_train, [self.batch_size,self.batch_size,3])
         tf.get_variable_scope().reuse_variables()
         print ("Building test network...............................\n")
         # text_seq has shape [T, N] and embedded_seq has shape [self.test_batch_size, T, D].
         embedded_seq_test = tf.nn.embedding_lookup(embedding_mat, text_feature_test)
         # 1. Encode the sentence into a vector representation, using the final
         # hidden states in a one-layer bidirectional LSTM network
         q_reshape_test = self.bilstm(embedded_seq_test)
         # 2. three attention units over the words in each sentence
          # 2. attention units over the words in each sentence fc(fc(Q+C+PRE+POST))
         q_reshape_flat = tf.reshape(q_reshape_test, [self.n_step_text * self.test_batch_size, self.n_hidden_text * 2])
         visual_feature_test=tf.transpose(visual_feature_test, [0, 2, 1])  # batch ctx fea
         visual_test=tf.reshape(visual_feature_test,[self.test_batch_size*(2*self.context_num+1),self.visual_feature_dim])
         query_term=fc('q2s_lt', q_reshape_flat, output_dim=self.semantic)
         moment_term=fc('c2s_lt', visual_test, output_dim=self.semantic)

         query_term=tf.reshape(query_term,[self.test_batch_size,self.n_step_text,self.semantic])
         moment_term=tf.reshape( moment_term,[self.test_batch_size,2*self.context_num+1,self.semantic])
         term2=tf.reduce_sum(moment_term,1,keep_dims=True)
         term2=tf.reshape(term2,[self.test_batch_size,self.semantic])
         term2=tf.reshape(tf.tile(term2,[1,self.n_step_text]),[self.test_batch_size,self.n_step_text,self.semantic])
         term=tf.nn.relu(tf.reshape(tf.add(query_term,term2), [self.n_step_text * self.test_batch_size, self.semantic]))
         scores_obj1 = fc('fc_scores_query_lt',term, output_dim=1)
         scores_obj1=tf.reshape(scores_obj1,[self.test_batch_size,self.n_step_text])
         is_not_pad=tf.cast(tf.not_equal(text_feature_test,0),tf.float32)
         #probs_obj1=tf.nn.softmax(scores_obj1)
         #probs_obj1=tf.mul(probs_obj1,is_not_pad)
         probs_obj1 = tf.mul(scores_obj1,is_not_pad)
         probs_obj1 = probs_obj1 / tf.reduce_sum(probs_obj1, 1, keep_dims=True)
         temp1= tf.transpose(tf.reshape(tf.tile(probs_obj1, [1, self.n_input_text]),[self.test_batch_size, self.n_input_text, self.n_step_text]),[0,2,1])  # [N,T,embed_dim]
         BoW_obj1 = tf.reduce_sum(tf.mul(temp1,embedded_seq_test), reduction_indices=1)
         print (BoW_obj1.get_shape().as_list())

         input_vision_obj1=tf.reshape(visual_feature_test,[self.test_batch_size,-1])
         # cross-modal part
         transformed_clip_test_norm = tf.nn.l2_normalize(input_vision_obj1 , dim=1)
         transformed_obj1_sent_test_norm = tf.nn.l2_normalize(BoW_obj1, dim=1)
         cross_modal_vec_test = self.cross_modal_comb(transformed_clip_test_norm,transformed_obj1_sent_test_norm, self.test_batch_size)  # batch batch 2*conmmon_space_dim
         sim_score_mat_test = vs_multilayer.vs_multilayer(cross_modal_vec_test, "vs_multilayer_lt",self.mpl_hidden,reuse=True)
         sim_score_mat_test = tf.reshape(sim_score_mat_test, [3])
       return sim_score_mat_train, sim_score_mat_test
    '''
    compute alignment and regression loss
    '''
    def compute_loss_reg(self, sim_reg_mat, offset_label):

        sim_score_mat, p_reg_mat, l_reg_mat = tf.split(2, 3, sim_reg_mat)
        sim_score_mat = tf.reshape(sim_score_mat, [self.batch_size, self.batch_size])
        l_reg_mat = tf.reshape(l_reg_mat, [self.batch_size, self.batch_size])
        p_reg_mat = tf.reshape(p_reg_mat, [self.batch_size, self.batch_size])
        # unit matrix with -2
        I_2 = tf.diag(tf.constant(-2.0, shape=[self.batch_size]))
        all1 = tf.constant(1.0, shape=[self.batch_size, self.batch_size])
        #               | -1  1   1...   |

        #   mask_mat =  | 1  -1  -1...   |

        #               | 1   1  -1 ...  |
        mask_mat = tf.add(I_2, all1)
        # loss cls, not considering iou
        I = tf.diag(tf.constant(1.0, shape=[self.batch_size]))
        #I_half = tf.diag(tf.constant(0.5, shape=[self.batch_size]))
        batch_para_mat = tf.constant(self.alpha, shape=[self.batch_size, self.batch_size])
        para_mat = tf.add(I,batch_para_mat)
        loss_mat = tf.log(tf.add(all1, tf.exp(tf.mul(mask_mat, sim_score_mat))))
        loss_mat = tf.mul(loss_mat, para_mat)
        loss_align = tf.reduce_mean(loss_mat)
        # regression loss
        l_reg_diag = tf.matmul(tf.mul(l_reg_mat, I), tf.constant(1.0, shape=[self.batch_size, 1]))
        p_reg_diag = tf.matmul(tf.mul(p_reg_mat, I), tf.constant(1.0, shape=[self.batch_size, 1]))
        offset_pred = tf.concat(1, (p_reg_diag, l_reg_diag))
        loss_reg = tf.reduce_mean(tf.abs(tf.sub(offset_pred, offset_label)))

        loss=tf.add(tf.mul(self.lambda_regression, loss_reg), loss_align)
        return loss, offset_pred, loss_reg


    def init_placeholder(self):
        visual_featmap_ph_train = tf.placeholder(tf.float32, shape=(self.batch_size, self.visual_feature_dim,2 * self.context_num + 1))  # input feature: current clip, pre-contex, and post contex
        sentence_ph_train = tf.placeholder(tf.int32, shape=(self.batch_size,self.n_step_text ))
        offset_ph = tf.placeholder(tf.float32, shape=(self.batch_size, 2))
        visual_featmap_ph_test = tf.placeholder(tf.float32, shape=(self.test_batch_size, self.visual_feature_dim,2 * self.context_num + 1))  # input feature: current clip, pre-contex, and post contex
        sentence_ph_test = tf.placeholder(tf.int32, shape=(self.test_batch_size,self.n_step_text ))

        return visual_featmap_ph_train, sentence_ph_train, offset_ph, visual_featmap_ph_test, sentence_ph_test

    def get_variables_by_name(self,name_list):
        v_list = tf.trainable_variables()
        v_dict = {}
        for name in name_list:
            v_dict[name] = []
        for v in v_list:
            for name in name_list:
                if name in v.name: v_dict[name].append(v)

        for name in name_list:
            print "Variables of <"+name+">"
            for v in v_dict[name]:
                print "    "+v.name
        return v_dict

    def training(self, loss):

        v_dict = self.get_variables_by_name(["lt"])
        vs_optimizer = tf.train.AdamOptimizer(self.vs_lr, name='vs_adam')
        vs_train_op = vs_optimizer.minimize(loss, var_list=v_dict["lt"])
        return vs_train_op


    def construct_model(self):
        # initialize the placeholder
        self.visual_featmap_ph_train, self.sentence_ph_train, self.offset_ph, self.visual_featmap_ph_test, self.sentence_ph_test=self.init_placeholder()
        # build inference network
        sim_reg_mat, sim_reg_mat_test= self.visual_semantic_infer(self.visual_featmap_ph_train, self.sentence_ph_train, self.visual_featmap_ph_test, self.sentence_ph_test)
        # compute loss
        self.loss_align_reg, offset_pred, loss_reg = self.compute_loss_reg(sim_reg_mat, self.offset_ph)
        # optimize
        self.vs_train_op = self.training(self.loss_align_reg)
        return self.loss_align_reg, self.vs_train_op, sim_reg_mat_test, offset_pred, loss_reg



