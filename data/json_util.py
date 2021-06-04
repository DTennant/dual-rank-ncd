import cv2 as cv
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import json



def get_img(path):
    img = Image.open(path).convert('RGB')
    
    return img

def get_sample_list_from_json(json_path):
    j = json.load(open(json_path, 'r'))
    samples = []
    for info in j['info_dicts']:
        try:
            samples.append((info['path'], info['label']))
        except KeyError:
            samples.append((info['path'], info['objects'][0]['label']))
    return samples


class JsonDataset(data.Dataset):
    def __init__(self, json_path, transform, target_transform=None, return_idx=True,
                 loader=get_img, parser=get_sample_list_from_json):
        self.json_path = json_path
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.parser = parser
        self.return_idx = return_idx

        self.samples = self.parser(self.json_path)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        img = self.loader(img)
        
        img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)

        if self.return_idx:
            return img, label, idx
        else:
            return img, label


    def __len__(self):
        return len(self.samples)



