import io
import numpy as np
import os
import random
import pickle
import re
PAD_IDENTIFIER = '<pad>'
UNK_IDENTIFIER = '<unk>' # <unk> is the word used to identify unknown words
SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')
wordembed_params = './word_embedding/embed_matrix.npy'
embedding_mat = np.load(wordembed_params)
vocab_file = './word_embedding/vocabulary_72700.txt'
T=10
'''
calculate temporal intersection over union
'''
def calculate_IoU(i0, i1):
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
    iou = 1.0*(inter[1]-inter[0])/(union[1]-union[0])
    return iou

'''
calculate the non Intersection part over Length ratia, make sure the input IoU is larger than 0
'''
def calculate_nIoL(base, sliding_clip):
    inter = (max(base[0], sliding_clip[0]), min(base[1], sliding_clip[1]))
    inter_l = inter[1]-inter[0]
    length = sliding_clip[1]-sliding_clip[0]
    nIoL = 1.0*(length-inter_l)/length
    return nIoL
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

class TrainingDataSet(object):
    def __init__(self, it_path, batch_size,sliding_dir,):

        self.batch_size = batch_size
        self.context_size = 128
        self.context_num = 1
        self.visual_feature_dim=4096
        self.clip_sentence_pairs_iou=pickle.load(open(it_path))
        self.sliding_clip_path=sliding_dir
        self.num_samples_iou=len(self.clip_sentence_pairs_iou)
        print str(len(self.clip_sentence_pairs_iou))+" iou clip-sentence pairs are readed"

    '''
    compute left (pre) and right (post) context features
    '''
    def get_context_window(self, clip_name, win_length):
        movie_name = clip_name.split("_")[0]
        start = int(clip_name.split("_")[1])
        end = int(clip_name.split("_")[2].split('.')[0])
        clip_length = self.context_size
        left_context_feats = np.zeros([win_length, 4096], dtype=np.float32)
        right_context_feats = np.zeros([win_length, 4096], dtype=np.float32)
        last_left_feat = np.load(self.sliding_clip_path+movie_name+'/'+clip_name)
        last_right_feat = np.load(self.sliding_clip_path+movie_name+'/'+clip_name)
        for k in range(win_length):
            left_context_start = start-clip_length*(k+1)
            left_context_end = start-clip_length*k
            right_context_start = end+clip_length*k
            right_context_end = end+clip_length*(k+1)
            left_context_name = movie_name+"_{0:06d}_{1:06d}.npy".format(left_context_start,left_context_end)
            right_context_name = movie_name+"_{0:06d}_{1:06d}.npy".format(right_context_start,right_context_end)
            if os.path.exists(self.sliding_clip_path+movie_name+'/'+left_context_name):
                left_context_feat = np.load(self.sliding_clip_path+movie_name+'/'+left_context_name)
                last_left_feat = left_context_feat
            else:
                left_context_feat = last_left_feat
            if os.path.exists(self.sliding_clip_path+movie_name+'/'+right_context_name):
                right_context_feat = np.load(self.sliding_clip_path+movie_name+'/'+right_context_name)
                last_right_feat = right_context_feat
            else:
                right_context_feat = last_right_feat
            left_context_feats[k] = left_context_feat
            right_context_feats[k] = right_context_feat
        return left_context_feats, right_context_feats


    '''
    read next batch of training data, this function is used for training CTRL-reg
    '''
    def next_batch_iou(self):

        random_batch_index = random.sample(range(self.num_samples_iou), self.batch_size)
        image_batch = np.zeros([self.batch_size, self.visual_feature_dim, 2 * self.context_num + 1])
        text_seq_batch = []
        offset_batch = np.zeros([self.batch_size, 2], dtype=np.float32)
        index = 0
        clip_set = set()
        while index < self.batch_size:
            k = random_batch_index[index]
            clip_name = self.clip_sentence_pairs_iou[k][0]
            if not clip_name in clip_set:
                clip_set.add(clip_name)
                movie_name=clip_name.split('_')[0]
                feat_path = self.sliding_clip_path+movie_name+'/'+self.clip_sentence_pairs_iou[k][2]
                featmap = np.load(feat_path)
                # read context features
                left_context_feat, right_context_feat = self.get_context_window(self.clip_sentence_pairs_iou[k][2], self.context_num)
                left_context_feat = np.reshape(left_context_feat, [self.visual_feature_dim])
                right_context_feat = np.reshape(right_context_feat, [self.visual_feature_dim])
                image_batch[index, :, :] = np.column_stack((left_context_feat, featmap, right_context_feat))
                text_seq_batch.append(preprocess_sentence(self.clip_sentence_pairs_iou[k][1], vocab_dict, T))
                p_offset = self.clip_sentence_pairs_iou[k][3]
                l_offset = self.clip_sentence_pairs_iou[k][4]
                offset_batch[index, 0] = p_offset
                offset_batch[index, 1] = l_offset
                index += 1
            else:
                r = random.choice(range(self.num_samples_iou))
                random_batch_index[index] = r
                continue

        return image_batch, text_seq_batch, offset_batch


class TestingDataSet(object):
    def __init__(self, csv_path, batch_size,img_dir ):
        #il_path: image_label_file path
        #self.index_in_epoch = 0
        #self.epochs_completed = 0
        self.batch_size = batch_size
        self.image_dir = img_dir
        self.visual_feature_dim=4096
        print "Reading testing data list from "
        self.sliding_clip_path = img_dir
        self.clip_sentence_pairs=pickle.load(open(csv_path))

        movie_names_set = set()
        self.movie_clip_names = {}
        for k in range(len(self.clip_sentence_pairs)):
            clip_name = self.clip_sentence_pairs[k][0]
            movie_name = clip_name.split("_")[0]
            if not movie_name in movie_names_set:
                movie_names_set.add(movie_name)
                self.movie_clip_names[movie_name] = []
            self.movie_clip_names[movie_name].append(k)
        self.movie_names = list(movie_names_set)

        self.clip_num_per_movie_max = 0
        for movie_name in self.movie_clip_names:
            if len(self.movie_clip_names[movie_name])>self.clip_num_per_movie_max: self.clip_num_per_movie_max = len(self.movie_clip_names[movie_name])
        print "Max number of clips in a movie is "+str(self.clip_num_per_movie_max)




    def get_context_window(self, clip_name, win_length):
        movie_name = clip_name.split("_")[0]
        start = int(clip_name.split("_")[1])
        end = int(clip_name.split("_")[2].split('.')[0])
        clip_length = 128#end-start
        left_context_feats = np.zeros([win_length,4096], dtype=np.float32)
        right_context_feats = np.zeros([win_length,4096], dtype=np.float32)
        last_left_feat = np.load(self.sliding_clip_path+movie_name+'/'+clip_name)
        last_right_feat = np.load(self.sliding_clip_path+movie_name+'/'+clip_name)
        for k in range(win_length):
            left_context_start = start-clip_length*(k+1)
            left_context_end = start-clip_length*k
            right_context_start = end+clip_length*k
            right_context_end = end+clip_length*(k+1)
            left_context_name = movie_name+"_{0:06d}_{1:06d}.npy".format(left_context_start,left_context_end)
            right_context_name = movie_name+"_{0:06d}_{1:06d}.npy".format(right_context_start,right_context_end)
            if os.path.exists(self.sliding_clip_path+movie_name+'/'+left_context_name):
                left_context_feat = np.load(self.sliding_clip_path+movie_name+'/'+left_context_name)
                last_left_feat = left_context_feat
            else:
                left_context_feat = last_left_feat
            if os.path.exists(self.sliding_clip_path+movie_name+'/'+right_context_name):
                right_context_feat = np.load(self.sliding_clip_path+movie_name+'/'+right_context_name)
                last_right_feat = right_context_feat
            else:
                right_context_feat = last_right_feat
            left_context_feats[k] = left_context_feat
            right_context_feats[k] = right_context_feat

        return left_context_feats,right_context_feats

    def load_movie_slidingclip(self, movie_name, sample_num):
        movie_clip_sentences = []
        movie_clip_featmap = []

        for k in range(len(self.clip_sentence_pairs)):
            if movie_name in self.clip_sentence_pairs[k][0]:
                movie_clip_sentences.append((self.clip_sentence_pairs[k][0], self.clip_sentence_pairs[k][1]))
        sliding_clip_names=os.listdir(self.sliding_clip_path+movie_name)
        for k in range(len(sliding_clip_names)):
            if "npy" in sliding_clip_names[k]:
              if movie_name in sliding_clip_names[k]:
                # print str(k)+"/"+str(len(self.movie_clip_names[movie_name]))
                visual_feature_path = self.sliding_clip_path+movie_name+'/'+sliding_clip_names[k]
                #context_feat=self.get_context(self.sliding_clip_names[k]+".npy")
                left_context_feat,right_context_feat = self.get_context_window(sliding_clip_names[k],1)
                feature_data = np.load(visual_feature_path)
                left_context_feat=np.reshape(left_context_feat,[self.visual_feature_dim])
                right_context_feat=np.reshape(right_context_feat,[self.visual_feature_dim])

                #comb_feat=np.hstack((context_feat,feature_data))
                comb_feat = np.column_stack((left_context_feat,feature_data,right_context_feat))
                movie_clip_featmap.append((sliding_clip_names[k], comb_feat))
        return movie_clip_featmap, movie_clip_sentences


