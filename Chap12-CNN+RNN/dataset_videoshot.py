# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 12:11:28 2020

@author: Jinsung
"""

import sys
sys.path.insert(0, 'C:\\Users\\Jinsung\\Documents\\Deep_Learning_Code\\Chap5-Classification Flower')
from dataset import *

class VideoShotDataset(Dataset):
    def __init__(self, filenames, timesteps=5):
        super(VideoShotDataset, self).__init__('videoshot', 'binary')
        
        video_create_cathe(filenames)
        
        self.frames, self.marks = video_load_cache(filenames)
        
        self.set_timesteps(timesteps)
        
    def set_timesteps(self, timesteps):
        self.timesteps = timesteps
        self.input_shape = [timesteps+1, 90, 120, 3]
        self.output_shape = [timesteps+1, 1]
        
    @property
    def train_count(self):
        return 2000
    
    def __str__(self):
        return '{}({}, {} frames, {} shots, {} train_data)'.format(self.name, self.mode, len(self.frames), np.sum(self.shots), self.train_count)
    
def video_create_cathe(filenames):
    movie_path = 'C:\\Users\\Jinsung\\Documents\\Deep_Learning_Code\\Datasets\\chap12\\movie\\'
    cache_path = 'C:\\Users\\Jinsung\\Documents\\Deep_Learning_Code\\Datasets\\chap12\\cache\\'
    
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
        
    for filename in filenames:
        movie_fname = movie_path + filename
        cache_fname = cache_path + filename + '.npy'
        
        print('{}: creating cache file...'.format(filename))
    
        cap = cv2.VideoCapture(movie_fname)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    
        shot_idxs = list(np.sort(np.random.randint(0, frame_count-400, 100)))
        thumbs = np.zeros([100,4,90,120,3])
        sn = 0 
    
        for fn in range(frame_count-400):
            ret = cap.grab()
            if fn == shot_idxs[sn]:
                for k in range(4):
                    _, frame = cap.retrieve(0)
                    cap.grab()
                    thumbs[sn, k] = cv2.resize(frame, (120, 90))
                sn += 1
                if sn >= 100:
                    break
        cap.release()
        np.save(cache_fname, thumbs)
        
    print('Creating thumbnail cache is done')

# 캐시 적재 함수의 정의
def video_load_cache(filenames):
    cache_path = 'C:\\Users\\Jinsung\\Documents\\Deep_Learning_Code\\Datasets\\chap12\\cache\\'
    
    buffer = np.zeros([len(filenames), 100, 4, 90, 120, 3])
    
    for n, filename in enumerate(filenames):
        cache_fname = cache_path + filename + '.npy'
        buffer[n] = np.load(cache_fname)
        
    starts = np.zeros([len(filenames), 100, 4])
    starts[:,:,0] = 1.0
    
    frames = buffer.reshape([-1, 90, 120, 3])
    shots = starts.reshape([-1])
    
    return frames, shots

# 데이터 공급 메서드 재정의
def video_shot_get_train_data(self, batch_size, nth):
    return self.create_seq(batch_size)

def video_shot_get_test_data(self):
    return self.create_seq(128)

def video_shot_get_validate_data(self, count):
    return self.create_seq(count)

VideoShotDataset.get_train_data = video_shot_get_train_data
VideoShotDataset.get_test_data = video_shot_get_test_data
VideoShotDataset.get_validate_data = video_shot_get_validate_data
VideoShotDataset.visualize_data = video_shot_get_validate_data

# 데이터 생성 메서드의 정의
def video_create_seq(self, count):
    length = self.timesteps
    xs = np.zeros([count, length+1, 90, 120, 3])
    ys = np.zeros([count, length+1, 1])
    frame_count = len(self.frames)
    for n in range(count):
        xs[n, 0, 0, 0, 0] = length
        ys[n, 0, 0] = length
        pos = frame_count
        for k in range(length):
            if pos >= frame_count-1 or np.random.randint(2) == 0:
                pos = np.random.randint(frame_count)
                is_new = 1.0
            else:
                pos += 1
                is_new = self.shots[pos]
            xs[n, k+1, :, :, :] = self.frames[pos, :, :, :]
            ys[n, k+1, 0] = is_new
    return xs, ys

VideoShotDataset.create_seq = video_create_seq

# 장면 전환 데이터셋을 위한 시각화 메서드 정의
def video_visualize(self, xs, est, ans):
    for n in range(len(xs)):
        draw_images_horz(xs[n][1:], [90,120,3])
        
    for n in range(len(xs)):
        print('Est: '+','.join(["%4.2f" % x for x in est[n,2:,0]]))
        print('Ans: '+','.join(["%4.2f" % x for x in ans[n,2:,0]]))
        
VideoShotDataset.visualize = video_visualize

# 시계열 이진 판단 출력의 후처리를 위한 순처리 및 역전파 메서드 재정의 
def video_shot_forward_postproc(self, output, y, mode=None):
    y1, o1 = y[:,2:,:], output[:,2:,:]
    entropy = sigmoid_cross_entropy_with_logits(y1, o1)
    loss = np.mean(entropy)
    aux = [y, output]
    
    return loss, aux

def video_shot_backprop_postproc(self, G_loss, aux, mode=None):
    y, output = aux
    
    y1, o1 = y[:,2:,:], output[:,2:,:]
    g_entropy = sigmoid_cross_entropy_with_logits_derv(y1, o1)
    G_entropy = g_entropy / np.prod(y1.shape)
    
    G_output = np.zeros(output.shape)
    G_output[:,0,:] = output[:,0,:]
    G_output[:,2:,:] = G_entropy
    
    return G_output

VideoShotDataset.forward_postproc = video_shot_forward_postproc
VideoShotDataset.backprop_postproc = video_shot_backprop_postproc

# 시계열 이진 판단 출력의 정확도와 추정치 변환 메서드 재정의
def video_shot_eval_accuracy(self, x, y, output, mode=None):
    y1, o1 = y[:,2:,:], output[:,2:,:]
    answer = np.equal(y1, 1.0)
    estimate = np.greater(o1, 0)
    correct = np.sum(np.equal(estimate, answer))
    accuracy = correct / np.prod(y1.shape)
    
    return accuracy

def video_shot_get_estimate(self, output, mode=None):
    estimate = np.zeros(output.shape)
    estimate[:,0,:] = output[:,0,:]
    est[:,2:,:] = sigmoid(output[:,2:,:])
    
    return estimate

VideoShotDataset.eval_accuracy = video_shot_eval_accuracy
VideoShotDataset.get_estimate = video_shot_get_estimate

