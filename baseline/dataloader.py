import torch
import torchvision
import torchtext
from torchtext import data
from torchtext import datasets
from torchtext.vacab import GloVe
import torch.nn as nn
import pickle


    def __init__(self, path, n_classes, train=True):
        if n_classes == 3:
            self.videoIDs, self.videoSpeakers, _, self.videoText,\
            self.videoAudio, self.videoSentence, self.trainVid,\
            self.testVid, self.videoLabels = pickle.load(open(path, 'rb'))
        elif n_classes == 7:
            self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
            self.videoAudio, self.videoSentence, self.trainVid,\
            self.testVid, _ = pickle.load(open(path, 'rb'))