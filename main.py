import tensorflow as tf
import numpy as np
import role_model
from six.moves import xrange
import time
import pickle
import operator
import os
import re
import io
os.environ["CUDA_VISIBLE_DEVICES"]="12"
PAD_IDENTIFIER = '<pad>'
UNK_IDENTIFIER = '<unk>' # <unk> is the word used to identify unknown words
SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')
wordembed_params = './word_embedding/embed_matrix.npy'
embedding_mat = np.load(wordembed_params)
vocab_file = './word_embedding/vocabulary_72700.txt'
T = 10
def load_vocab_dict_from_file(dict_file, pad_at_first=True):
            with io.open(dict_file, encoding='utf-8') as f:
                words = [w.strip() for w in f.readlines()]
            if pad_at_first and words[0] != '<pad>':
                raise Exception("The first word needs to be <pad> in the word list.")
            vocab_dict = {words[n]: n for n in range(len(words))}
            return vocab_dict

def sentence2vocab_indices(sentence, vocab_dict):
            if isinstance(sentence, bytes):
                sentence = sentence.decode()
            words = SENTENCE_SPLIT_REGEX.split(sentence.strip())
            words = [w.lower() for w in words if len(w.strip()) > 0]
            # remove .
            if len(words) > 0 and (words[-1] == '.' or words[-1] == '?'):
                words = words[:-1]
            vocab_indices = [(vocab_dict[w] if w in vocab_dict else vocab_dict[UNK_IDENTIFIER])
                             for w in words]
            return vocab_indices

def preprocess_vocab_indices(vocab_indices, vocab_dict, T):
            # Truncate long sentences
            if len(vocab_indices) > T:
                vocab_indices = vocab_indices[:T]
            # Pad short sentences at the beginning with the special symbol '<pad>'
            if len(vocab_indices) < T:
                vocab_indices = [vocab_dict[PAD_IDENTIFIER]] * (T - len(vocab_indices)) + vocab_indices
            return vocab_indices

def preprocess_sentence(sentence, vocab_dict, T):
            vocab_indices = sentence2vocab_indices(sentence, vocab_dict)
            return preprocess_vocab_indices(vocab_indices, vocab_dict, T)

vocab_dict = load_vocab_dict_from_file(vocab_file)
def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset+labels_dense.ravel()] = 1
    return labels_one_hot

def calculate_IoU(i0,i1):
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
    iou = 1.0*(inter[1]-inter[0])/(union[1]-union[0])
    return iou

def nms_temporal(x1,x2,s, overlap):
    pick = []
    assert len(x1)==len(s)
    assert len(x2)==len(s)
    if len(x1)==0:
        return pick

    #x1 = [b[0] for b in boxes]
    #x2 = [b[1] for b in boxes]
    #s = [b[-1] for b in boxes]
    union = map(operator.sub, x2, x1) # union = x2-x1
    I = [i[0] for i in sorted(enumerate(s), key=lambda x:x[1])] # sort and get index

    while len(I)>0:
        i = I[-1]
        pick.append(i)

        xx1 = [max(x1[i],x1[j]) for j in I[:-1]]
        xx2 = [min(x2[i],x2[j]) for j in I[:-1]]
        inter = [max(0.0, k2-k1) for k1, k2 in zip(xx1, xx2)]
        o = [inter[u]/(union[i] + union[I[u]] - inter[u]) for u in range(len(I)-1)]
        I_new = []
        for j in range(len(o)):
            if o[j] <=overlap:
                I_new.append(I[j])
        I = I_new
    return pick

'''
compute recall at certain IoU
'''
def compute_IoU_recall_top_n_forreg(top_n, iou_thresh, sentence_image_mat, sentence_image_reg_mat, sclips, iclips,movie_clip_sentences):
    correct_num = 0.0
    for k in range(sentence_image_mat.shape[0]):
        gt = sclips[k]
        gt_start = float(gt.split("_")[1])
        gt_end = float(gt.split("_")[2].split('.')[0])
        #print gt +" "+str(gt_start)+" "+str(gt_end)
        sim_v = [v for v in sentence_image_mat[k]]
        starts = [s for s in sentence_image_reg_mat[k,:,0]]
        ends = [e for e in sentence_image_reg_mat[k,:,1]]
        picks = nms_temporal(starts,ends, sim_v, iou_thresh-0.05)
        if top_n<len(picks): picks=picks[0:top_n]
        for index in picks:
            pred_start = sentence_image_reg_mat[k, index, 0]
            pred_end = sentence_image_reg_mat[k, index, 1]
            iou = calculate_IoU((gt_start, gt_end),(pred_start, pred_end))
            if iou>=iou_thresh:
                correct_num+=1
                break
    return correct_num

'''
evaluate the model
'''
def do_eval_slidingclips(sess, vs_eval_op, model, movie_length_info, iter_step,context_num):
    IoU_thresh = [0.1,0.3, 0.5,0.7,0.9]
    all_correct_num_10 = [0.0]*5
    all_correct_num_5 = [0.0]*5
    all_correct_num_1 = [0.0]*5
    all_retrievd = 0.0
    for movie_name in model.test_set.movie_names:
        print "Test movie: "+movie_name+"....loading movie data"
        movie_clip_featmaps, movie_clip_sentences=model.test_set.load_movie_slidingclip(movie_name, 16)
        print "sentences: "+ str(len(movie_clip_sentences))
        print "clips: "+ str(len(movie_clip_featmaps))
        sentence_image_mat=np.zeros([len(movie_clip_sentences), len(movie_clip_featmaps)])
        sentence_image_reg_mat=np.zeros([len(movie_clip_sentences), len(movie_clip_featmaps), 2])
        for k in range(len(movie_clip_sentences)):
           
            sent_vec=movie_clip_sentences[k][1]
            sent_vec = preprocess_sentence(sent_vec, vocab_dict, T)
            sent_vec=np.reshape(sent_vec,[1,T])
            for t in range(len(movie_clip_featmaps)):
                featmap = movie_clip_featmaps[t][1]
                visual_clip_name = movie_clip_featmaps[t][0]
                start = float(visual_clip_name.split("_")[1])
                end = float(visual_clip_name.split("_")[2].split(".")[0])
                featmap = np.reshape(featmap, [ 1, featmap.shape[0],2*context_num+1]) #batch, 4096, 3
                feed_dict = {
                model.visual_featmap_ph_test: featmap,
                model.sentence_ph_test:sent_vec
                }
                outputs = sess.run(vs_eval_op,feed_dict=feed_dict)
                sentence_image_mat[k,t] = outputs[0]
                reg_end = end+outputs[2]
                reg_start = start+outputs[1]

                sentence_image_reg_mat[k,t,0] = reg_start
                sentence_image_reg_mat[k,t,1] = reg_end
        iclips = [b[0] for b in movie_clip_featmaps]
        sclips = [b[0] for b in movie_clip_sentences]

        # calculate Recall@m, IoU=n
        for k in range(len(IoU_thresh)):
            IoU=IoU_thresh[k]
            correct_num_10 = compute_IoU_recall_top_n_forreg(10, IoU, sentence_image_mat, sentence_image_reg_mat, sclips, iclips,movie_clip_sentences)
            correct_num_5 = compute_IoU_recall_top_n_forreg(5, IoU, sentence_image_mat, sentence_image_reg_mat, sclips, iclips,movie_clip_sentences)
            correct_num_1 = compute_IoU_recall_top_n_forreg(1, IoU, sentence_image_mat, sentence_image_reg_mat, sclips, iclips,movie_clip_sentences)
            print movie_name+" IoU="+str(IoU)+", R@10: "+str(correct_num_10/len(sclips))+"; IoU="+str(IoU)+", R@5: "+str(correct_num_5/len(sclips))+"; IoU="+str(IoU)+", R@1: "+str(correct_num_1/len(sclips))
            all_correct_num_10[k]+=correct_num_10
            all_correct_num_5[k]+=correct_num_5
            all_correct_num_1[k]+=correct_num_1
        all_retrievd+=len(sclips)
    for k in range(len(IoU_thresh)):
        print " IoU="+str(IoU_thresh[k])+", R@10: "+str(all_correct_num_10[k]/all_retrievd)+"; IoU="+str(IoU_thresh[k])+", R@5: "+str(all_correct_num_5[k]/all_retrievd)+"; IoU="+str(IoU_thresh[k])+", R@1: "+str(all_correct_num_1[k]/all_retrievd)
        

def run_training():
    max_steps=100000
    lr=0.01
    current_lr=lr
    train_batch_size=50
    display_step=5
    test_iter=50000
    semantic_size=1024
    mpl_hidden=1000
    context_num=1
    visual_feature_dim=4096
    embed_dim = 300
    lstm_dim = 1000
    train_csv_path = "./train_iou_clip_sentence_pairs.pkl"
    test_csv_path = "./test_clip-sentvec_charades.pkl"
    test_feature_dir="/mnt/sata/meng/charades_test/"
    train_feature_dir = "/mnt/sata/meng/charades_train/"
    model = role_model.ROLE_Model(train_batch_size, train_csv_path, test_csv_path , lr, visual_feature_dim, embed_dim,lstm_dim, T,train_feature_dir,test_feature_dir,semantic_size,mpl_hidden)
    with tf.Graph().as_default():

        loss_align_reg, vs_train_op, vs_eval_op, offset_pred, loss_reg = model.construct_model()
        # Create a session for running Ops on the Graph.
        sess = tf.Session()
        # Run the Op to initialize the variables.
        init = tf.initialize_all_variables()
        sess.run(init)
        for step in xrange(max_steps):
            start_time = time.time()
            feed_dict = model.fill_feed_dict_train_reg()
            _, loss_value, offset_pred_v, loss_reg_v = sess.run([vs_train_op, loss_align_reg, offset_pred, loss_reg], feed_dict=feed_dict)
            duration = time.time() - start_time

            if step % display_step == 0:
                # Print status to stdout.
                print('Step %d: loss = %.3f (%.3f sec)' % (step, loss_value, duration))
            if (step+1)%50000==0:
                current_lr=current_lr*0.1
                model.vs_lr=current_lr
                print (model.vs_lr)
            if (step+1) % test_iter == 0:
                print "Start to test:-----------------\n"
                movie_length_info=pickle.load(open("./video_allframes_info_charades.pkl"))
                do_eval_slidingclips(sess, vs_eval_op, model, movie_length_info, step+1,context_num)

def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()




