import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import pandas as pd


class MELDDataset(Dataset): #make Dataset of MELD
    def __init__(self, path, n_classes, train=True):
        if n_classes == 3:
            self.videoIDs, self.videoSpeakers, _, self.videoText,\
            self.videoAudio, self.videoSentence, self.trainVid,\
            self.testVid, self.videoLabels = pickle.load(open(path, 'rb'))
            # label index mapping = {'neutral': 0, 'positive': 1, 'negative': 2}
        elif n_classes == 7:
            self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
            self.videoAudio, self.videoSentence, self.trainVid,\
            self.testVid, _ = pickle.load(open(path, 'rb'))
         # label index mapping = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}

         self.keys = [x for x in (self.trainVid if train else self.testVid)] #divide trainVid & testVid
         self.len = len(self.keys) # the number of Vid

    def __len__(self):
        return self.len  # num of sample

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch FloatTensor(self.videoText[vid]), \
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor(self.videoSpeakers[vid]),\
               torch.FloatTensor([1]*len(len(self.videoLabels[vid]))),\
               torch.LongTensor(self.videoLabels[vid]),\
               vid

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<3 else pad_sequence(dat[i], True) if i<5 else dat[i].tolist() for i in dat] #collate data size






