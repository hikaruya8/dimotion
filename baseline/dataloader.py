import torch
import torchvision
import torchtext
from torchtext import data
from torchtext import datasets
from torchtext.vacab import GloVe
import torch.nn as nn
import pickle


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

    def __getitem__():

