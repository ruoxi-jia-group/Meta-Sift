from ast import Not
import logging
import os
from models import ResNet18
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
import cv2 as cv
import torch.nn as nn
from collections import OrderedDict
import copy
from PIL import Image
from tqdm import tqdm
import random
from torch.autograd import Variable

seed = 0

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class my_subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """
    def __init__(self, dataset, indices,  transform):
        self.indices = indices
        if not isinstance(self.indices, list):
            self.indices = list(self.indices)
        self.dataset = Subset(dataset, self.indices)
        self.transform = transform

    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        label = self.dataset[idx][1]
        if self.transform != None:
            # image = Image.fromarray(image.astype(np.uint8))
            image = self.transform(image)
        return (image, label)

    def __len__(self):
        return len(self.indices)

class delete_subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """
    def __init__(self, dataset, indices):
        self.indices = indices
        if not isinstance(self.indices, list):
            self.indices = list(self.indices)
        self.data = copy.deepcopy(self.dataset.data) 
        self.targets = copy.deepcopy(self.dataset.targets)
        self.data = np.delete(self.data,indices,0)
        self.targets = np.delete(self.targets,indices)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]
        return (image, label)

    def __len__(self):
        return len(self.targets)
        

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)
            
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target
    
def apply_noise_patch(noise,images,offset_x=0,offset_y=0,mode='change',padding=20,position='fixed'):
    '''
    noise: torch.Tensor(1, 3, pat_size, pat_size)
    images: torch.Tensor(N, 3, 512, 512)
    outputs: torch.Tensor(N, 3, 512, 512)
    '''
    length = images.shape[2] - noise.shape[2]
    if position == 'fixed':
        wl = offset_x
        ht = offset_y
    else:
        wl = np.random.randint(padding,length-padding)
        ht = np.random.randint(padding,length-padding)
    if len(images.shape) == 3:
        noise_now = np.copy(noise[0,:,:,:])
        wr = length-wl
        hb = length-ht
        m = nn.ZeroPad2d((wl, wr, ht, hb))
        if(mode == 'change'):
            images[:,ht:ht+noise.shape[2],wl:wl+noise.shape[3]] = 0
            images[:,ht:ht+noise.shape[2],wl:wl+noise.shape[3]] = noise_now
        else:
            images += noise_now
    else:
        for i in range(images.shape[0]):
            noise_now = np.copy(noise)
            wr = length-wl
            hb = length-ht
            m = nn.ZeroPad2d((wl, wr, ht, hb))
            if(mode == 'change'):
                images[i:i+1,:,ht:ht+noise.shape[2],wl:wl+noise.shape[3]] = 0
                images[:,ht:ht+noise.shape[2],wl:wl+noise.shape[3]] = noise_now
            else:
                images[i:i+1] += noise_now
    return images

class noisy_label(Dataset):
    def __init__(self, dataset, indices, num_classes, transform, seed):
        set_seed(seed)
        print('Random seed is: ', seed)
        self.dataset = dataset
        self.indices = indices
        self.data = copy.deepcopy(self.dataset.data) 
        self.targets = copy.deepcopy(self.dataset.targets)
        self.num_classes = num_classes
        self.transform = transform

        allos_idx = []
        for i in range(num_classes):
            allowed_values = list(range(num_classes))
            allowed_values.remove(i)
            allos_idx.append(allowed_values)
        for i in range(len(indices)):
            tar_lab = self.targets[indices[i]]
            self.targets[indices[i]] = random.choice(allos_idx[tar_lab])

    def __getitem__(self, idx):
        image = self.data[idx]
        target = self.targets[idx]
        if self.transform is not None:
            image = Image.fromarray(image.astype(np.uint8))
            image = self.transform(image)
        return (image, target)

    def __len__(self):
        return len(self.dataset)

class flipping_label(Dataset):
    def __init__(self, dataset, indices, tar_lab, transform, seed):
        set_seed(seed)
        self.dataset = dataset
        self.indices = indices
        self.data = copy.deepcopy(self.dataset.data) 
        self.targets = copy.deepcopy(self.dataset.targets)
        self.tar_lab = tar_lab
        for i in self.indices:
            self.targets[i] = self.tar_lab
        self.transform = transform

    def __getitem__(self, idx):
        image = self.data[idx]
        target = self.targets[idx]
        if self.transform is not None:
            image = Image.fromarray(image.astype(np.uint8))
            image = self.transform(image)
        return (image, target)

    def __len__(self):
        return len(self.dataset)

class change_label(Dataset):
    def __init__(self, dataset, tar_lab):
        set_seed(seed)
        self.dataset = dataset
        self.data = copy.deepcopy(self.dataset.data) 
        self.targets = copy.deepcopy(self.dataset.targets)
        self.indices = np.where(np.array(self.targets)==tar_lab[0])[0]
        self.tar_lab = tar_lab

    def __getitem__(self, idx):
        image = self.dataset[self.indices[idx]][0]
        return (image, self.tar_lab[1])

    def __len__(self):
        return self.indices.shape[0]

class posion_image_nottar_label(Dataset):
    def __init__(self, dataset,indices,noise,lab):
        self.dataset = dataset
        self.indices = indices
        self.noise = noise
        self.lab = lab

    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        label = self.dataset[idx][1]
        if idx in self.indices:
            image = torch.clamp(apply_noise_patch(self.noise,image,mode='add'),-1,1)
            label = self.lab
        return (image, label)

    def __len__(self):
        return len(self.dataset)

class posion_image_all2all(Dataset):
    def __init__(self, dataset,noise,poi_list,num_classes, transform):
        self.dataset = dataset
        self.data = dataset.data
        self.targets = self.dataset.targets
        self.poi_list = poi_list
        self.noise = noise
        self.num_classes = num_classes
        self.transform = transform
        for i in self.poi_list:
            self.targets[i] = self.targets[i] + 1
            if self.targets[i] == self.num_classes:
                self.targets[i] = 0

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]
        if idx in self.poi_list:
            pat_size = 4
            image[32 - 1 - pat_size:32 - 1, 32 - 1 - pat_size:32 - 1, :] = 255
        
        if self.transform is not None:
            image = Image.fromarray(image.astype(np.uint8))
            image = self.transform(image)
        return (image, label)

    def __len__(self):
        return len(self.dataset)

class posion_noisy_all2all(Dataset):
    def __init__(self, dataset,noise,poi_list,num_classes, transform):
        self.dataset = dataset
        self.data = copy.deepcopy(self.dataset.data) 
        self.targets = copy.deepcopy(self.dataset.targets)
        self.poi_list = poi_list
        self.noise = noise
        self.num_classes = num_classes
        self.transform = transform
        for i in self.poi_list:
            self.targets[i] = self.targets[i] + 1
            if self.targets[i] == self.num_classes:
                self.targets[i] = 0

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]
        if idx in self.poi_list:
            image = image.astype(int)
            image += self.noise
            image = np.clip(image,0,255)
        
        if self.transform is not None:
            image = Image.fromarray(image.astype(np.uint8))
            image = self.transform(image)
        return (image, label)

    def __len__(self):
        return len(self.dataset)

class posion_image_all2one(Dataset):
    def __init__(self, dataset,poi_list,tar_lab, transform):
        self.dataset = dataset
        self.data = copy.deepcopy(self.dataset.data) 
        self.targets = copy.deepcopy(self.dataset.targets)
        self.poi_list = poi_list
        self.tar_lab = tar_lab
        self.transform = transform
        for i in self.poi_list:
            self.targets[i] = self.tar_lab

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]
        if idx in self.poi_list:
            pat_size = 4
            image[32 - 1 - pat_size:32 - 1, 32 - 1 - pat_size:32 - 1, :] = 255
        if self.transform is not None:
            image = Image.fromarray(image.astype(np.uint8))
            image = self.transform(image)
        return (image, label)

    def __len__(self):
        return len(self.dataset)

class posion_noisy_all2one(Dataset):
    def __init__(self, dataset,poi_list,tar_lab, transform, noisy):
        self.dataset = dataset
        self.data = copy.deepcopy(self.dataset.data) 
        self.targets = copy.deepcopy(self.dataset.targets)
        self.poi_list = poi_list
        self.tar_lab = tar_lab
        self.transform = transform
        self.noisy = noisy
        for i in self.poi_list:
            self.targets[i] = self.tar_lab

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]
        if idx in self.poi_list:
            image = image.astype(int)
            image += self.noisy
            image = np.clip(image,0,255)
        if self.transform is not None:
            image = Image.fromarray(image.astype(np.uint8))
            image = self.transform(image)
        return (image, label)

    def __len__(self):
        return len(self.dataset)

class posion_image(Dataset):
    def __init__(self, dataset,indices,noise, transform):
        self.dataset = dataset
        self.indices = indices
        self.noise = noise
        self.data = copy.deepcopy(self.dataset.data) 
        self.targets = copy.deepcopy(self.dataset.targets)
        self.transform = transform

    def __getitem__(self, idx):
        image = self.data[idx]
        if idx in self.indices:
            image = image.astype(int)
            image += self.noise
            image = np.clip(image,0,255)
        if self.transform is not None:
            image = Image.fromarray(image.astype(np.uint8))
            image = self.transform(image)
        label = self.targets[idx]
        return (image, label)

    def __len__(self):
        return len(self.dataset)
    
class posion_image_label(Dataset):
    def __init__(self, dataset,indices,noise,target,transform):
        self.dataset = dataset
        self.indices = indices
        self.noise = noise
        self.target = target
        self.targets = copy.deepcopy(self.dataset.targets)
        self.data = copy.deepcopy(self.dataset.data) 
        self.transform = transform

    def __getitem__(self, idx):
        image = self.data[idx]
        if idx in self.indices:
            image = image.astype(int)
            image += self.noise
            image = np.clip(image,0,255)
        if self.transform is not None:
            image = Image.fromarray(image.astype(np.uint8))
            image = self.transform(image)
        #label = self.dataset[idx][1]
        return (image, self.target)

    def __len__(self):
        return len(self.indices)
    
class get_labels(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        return self.dataset[idx][1]

    def __len__(self):
        return len(self.dataset)

class concoct_dataset(torch.utils.data.Dataset):
    def __init__(self, target_dataset,outter_dataset):
        self.idataset = target_dataset
        self.odataset = outter_dataset

    def __getitem__(self, idx):
        if idx < len(self.odataset):
            img = self.odataset[idx][0]
            labels = self.odataset[idx][1]
        else:
            img = self.idataset[idx-len(self.odataset)][0]
            #labels = torch.tensor(len(self.odataset.classes),dtype=torch.long)
            labels = len(self.odataset.classes)
        #label = self.dataset[idx][1]
        return (img,labels)

    def __len__(self):
        return len(self.idataset)+len(self.odataset)

def inverse_normalize(img, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    for i in range(len(mean)):
        img[:,:,i] = img[:,:,i]*std[i]+mean[i]
    return img

def poi_dataset(Dataset, poi_methond='badnets', transform=None, tar_lab = 0, poi_rates = 0.2, random_seed = 0, noisy = None):
    set_seed(random_seed)
    label = Dataset.targets
    num_classes = len(np.unique(label))
    if poi_methond == 'backdoor_all2all':
        badnets_noise = np.zeros((1, 3, 32, 32))
        badnets_noise[0,:,26:31,26:31] = 255
        poi_idx = []
        for i in range(num_classes):
            current_label = np.where(np.array(label)==i)[0]
            samples_idx = np.random.choice(current_label, size=int(current_label.shape[0] * poi_rates), replace=False)
            poi_idx.extend(samples_idx)
        posion_dataset = posion_image_all2all(Dataset,badnets_noise, poi_idx, num_classes, transform)
        return posion_dataset, poi_idx
    elif poi_methond == 'noisy_label':
        poi_idx = []
        for i in range(num_classes):
            current_label = np.where(np.array(label)==i)[0]
            samples_idx = np.random.choice(current_label, size=int(current_label.shape[0] * poi_rates), replace=False)
            poi_idx.extend(samples_idx)
        posion_dataset = noisy_label(Dataset, poi_idx, num_classes, transform, random_seed)
        return posion_dataset, poi_idx
    elif poi_methond == 'flipping_label':
        poi_idx = []
        current_label = np.where(np.array(label)==tar_lab[0])[0]
        samples_idx = np.random.choice(current_label, size=int(current_label.shape[0] * poi_rates), replace=False)
        poi_idx.extend(samples_idx)
        posion_dataset = flipping_label(Dataset, poi_idx, tar_lab[1], transform, random_seed)
        return posion_dataset, poi_idx
    elif poi_methond == 'backdoor':
        current_label = np.where(np.array(label)!=tar_lab)[0]
        poi_idx = np.random.choice(current_label, size=int(len(Dataset) * poi_rates), replace=False)
        posion_dataset = posion_image_all2one(Dataset, poi_idx, tar_lab, transform)
        return posion_dataset, poi_idx
    elif poi_methond == 'clean_label_narcissus':
        current_label = np.where(np.array(label)==tar_lab)[0]
        poi_idx = np.random.choice(current_label, size=int(current_label.shape[0] * poi_rates), replace=False)
        if noisy is None:
            noisy = np.transpose(np.load('/home/minzhou/public_html/unlearnable/rebuttal/Narcissus-backdoor-attack/checkpoint/best_noise_gtsrb_mismatch_vgg_08-04-15_00_38.npy')[0],(1,2,0))
            # noisy = np.transpose(np.load('/home/minzhou/public_html/unlearnable/rebuttal/Narcissus-backdoor-attack/checkpoint/best_noise_cifar10_mismatch_06-09-05_39_13.npy')[0],(1,2,0))*0.5
            # noisy = ((inverse_normalize(noisy,(0.5, 0.5, 0.5),(0.5, 0.5, 0.5))-0.5)*255).astype(int)
            noisy = (noisy*255).astype(int)
        posion_dataset = posion_image(Dataset, poi_idx, noisy, transform)
        return posion_dataset, poi_idx
    elif poi_methond == 'noisy_all2one':
        current_label = np.where(np.array(label)!=tar_lab)[0]
        poi_idx = np.random.choice(current_label, size=int(len(Dataset) * poi_rates), replace=False)
        if noisy is None:
            noisy = (np.load('/home/minzhou/public_html/unlearnable/cifar10/best_universal.npy')[0]*255).astype(int)
        posion_dataset = posion_noisy_all2one(Dataset, poi_idx, tar_lab, transform, noisy)
        return posion_dataset, poi_idx
    if poi_methond == 'noisy_all2all':
        if noisy is None:
            noisy = (np.load('/home/minzhou/public_html/unlearnable/cifar10/best_universal.npy')[0]*255).astype(int)
        poi_idx = []
        for i in range(num_classes):
            current_label = np.where(np.array(label)==i)[0]
            samples_idx = np.random.choice(current_label, size=int(current_label.shape[0] * poi_rates), replace=False)
            poi_idx.extend(samples_idx)
        posion_dataset = posion_noisy_all2all(Dataset,noisy, poi_idx, num_classes, transform)
        return posion_dataset, poi_idx

import h5py
class h5_dataset(Dataset):
    def __init__(self, path, train, transform):
        f = h5py.File(path,'r') 
        if train:
            self.data = np.vstack((np.asarray(f['X_train']),np.asarray(f['X_val']))).astype(np.uint8)
            self.targets = list(np.argmax(np.vstack((np.asarray(f['Y_train']),np.asarray(f['Y_val']))),axis=1))
        else:
            self.data = np.asarray(f['X_test']).astype(np.uint8)
            self.targets = list(np.argmax(np.asarray(f['Y_test']),axis=1))
        self.transform = transform

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]
        if self.transform is not None:
            image = Image.fromarray(image.astype(np.uint8))
            image = self.transform(image)
        return (image, label)

    def __len__(self):
        return len(self.targets)

cfg = {'small_VGG16': [32, 32, 'M', 64, 64, 'M', 128, 128, 'M'],}
drop_rate = [0.3,0.4,0.4]

class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(2048, 43)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        key = 0
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2),
                           nn.Dropout(drop_rate[key])]
                key += 1
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ELU(inplace=True)]
#                            nn.ReLU(inplace=True)]
                in_channels = x
#         layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def norm_weight(weights):
    norm = torch.sum(weights)
    if norm >= 0.0001:
        normed_weights = weights / norm
    else:
        normed_weights = weights
    return normed_weights

class nnVent(nn.Module):
    def __init__(self, input, hidden1, hidden2, output):
        super(nnVent, self).__init__() 
        self.linear1 = nn.Linear(input, hidden1)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden1, hidden2)
        self.linear3 = nn.Linear(hidden2, output)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu1(x)
        out = self.linear3(x)
        return torch.sigmoid(out)

def build_training(args):
    model = ResNet18(num_classes = args.num_classes).cuda()
    optimizer_a = torch.optim.Adadelta(model.parameters(), args.lr,weight_decay=args.weight_decay)

    vnet = nnVent(1, 100, 150, 1).cuda()
    optimizer_c = torch.optim.RAdam(vnet.parameters(), args.meta_lr)
    return model, optimizer_a, vnet, optimizer_c

def warmup(model, optimizer, data_loader, args):
    for w_i in range(args.warmup_epochs):
        for iters, (input_train, target_train) in enumerate(data_loader):
            model.train()
            input_var,target_var = input_train.cuda(), target_train.cuda()
            optimizer.zero_grad()
            outputs = model(input_var)
            loss = F.cross_entropy(outputs, target_var)
            loss.backward()  
            optimizer.step() 
        print('Warmup Epoch {} '.format(w_i)) 
    return model, optimizer



def grad_function(grad, grad_model):
    grad_size = grad.size()
    if len(grad_size) == 4:
        reduced_grad = torch.sum(grad, dim=[1, 2, 3]).view(-1, grad_size[0])
        grad_act = grad_model(reduced_grad.detach())
        grad_act = grad_act[:, 1].view(-1)
    elif len(grad_size) == 2:
        reduced_grad = torch.sum(grad, dim=[1]).view(-1, grad_size[0])
        grad_act = grad_model(reduced_grad.detach())
        grad_act = grad_act[:, 1].view(-1)
    else:
        reduced_grad = grad.view(-1, grad_size[0])
        grad_act = grad_model(reduced_grad.detach())
        grad_act = grad_act[:, 1].view(-1)
    return grad_act

def compute_gated_grad(grads, grad_models, num_opt, num_act):
    new_grads = []
    acts = []
    gates = []
    for grad in grads[0:-num_opt]:
        new_grads.append(grad.detach())
    for g_id, grad in enumerate(grads[-num_opt:-2]):
        grad_act = grad_function(grad, grad_models[g_id])
        if grad_act > 0.5:
            new_grads.append(grad_act * grad)
        else:
            new_grads.append((1-grad_act) * grad.detach())
    acts.append(grad_act)
    for grad in grads[-2::]:
        new_grads.append(grad)
    act_loss = (torch.sum(torch.cat(acts)) - num_act)**2
    return new_grads, act_loss

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

class nnGradGumbelSoftmax(nn.Module):
    def __init__(self, input, hidden, input_norm=False):
        super(nnGradGumbelSoftmax, self).__init__()
        #self.bn = MetaBatchNorm1d(input)
        self.linear1 = nn.Linear(input, hidden)
        self.relu1 = nn.PReLU()
        self.linear2 = nn.Linear(hidden, hidden)
        self.relu2 = nn.PReLU()

        self.act = nn.Linear(hidden, 2)
        self.register_buffer('weight_act', to_var(self.act.weight.data, requires_grad=True))
        self.register_buffer('bias_act', to_var(self.act.bias.data, requires_grad=True))
        self.input_norm = input_norm
        
    def forward(self, x):
        if self.input_norm:
            x_mean, x_std = x.mean(), x.std()
            x = (x-x_mean)/(x_std+1e-9)
        x = self.relu1(self.linear1(x))
        x = self.relu2(self.linear2(x))
        x = F.linear(x, self.weight_act, self.bias_act)
        y = F.gumbel_softmax(x,tau=5)
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        return y_hard

def sample_gumbel(shape, eps=1e-20):
    U = torch.cuda.FloatTensor(shape).uniform_()
    return -Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature = 5):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y

def build_thres_model(args, weight_shape):
    hidden_dim = 128
    model = nnGradGumbelSoftmax(weight_shape[0], hidden_dim, input_norm=True)
    model.cuda()
    return model

def build_grad_models(args, model):
    grad_models = []
    grad_optimizers = []
    for param in list(model.parameters())[-args.top_k:-2]:
        param_shape = param.size()
        _grad_model = build_thres_model(args, param_shape)
        _optimizer = torch.optim.SGD(_grad_model.parameters(), args.go_lr,
            momentum=args.momentum, nesterov=args.nesterov,
            weight_decay=0)
        grad_models.append(_grad_model)
        grad_optimizers.append(_optimizer)
    return grad_models, grad_optimizers

from torch.optim.sgd import SGD


class MetaSGD(SGD):
    def __init__(self, net, *args, **kwargs):
        super(MetaSGD, self).__init__(*args, **kwargs)
        self.net = net

    def set_parameter(self, current_module, name, parameters):
        if '.' in name:
            name_split = name.split('.')
            module_name = name_split[0]
            rest_name = '.'.join(name_split[1:])
            for children_name, children in current_module.named_children():
                if module_name == children_name:
                    self.set_parameter(children, rest_name, parameters)
                    break
        else:
            # current_module._parameters[name] = parameters
            current_module._parameters.__setitem__(name, parameters)

    def meta_step(self, grads):
        group = self.param_groups[0]
        # weight_decay = group['weight_decay']
        # momentum = group['momentum']
        # dampening = group['dampening']
        # nesterov = group['nesterov']
        lr = group['lr']


        for (name, parameter), grad in zip(self.net.named_parameters(), grads):
            parameter.detach_()
            # if weight_decay != 0:
            #     grad_wd = grad.add(parameter, alpha=weight_decay)
            # else:
            #     grad_wd = grad
            # if momentum != 0 and 'momentum_buffer' in self.state[parameter]:
            #     buffer = self.state[parameter]['momentum_buffer']
            #     grad_b = buffer.mul(momentum).add(grad_wd, alpha=1-dampening)
            # else:
            #     grad_b = grad_wd
            # if nesterov:
            #     grad_n = grad_wd.add(grad_b, alpha=momentum)
            # else:
            #     grad_n = grad_b
            self.set_parameter(self.net, name, parameter.add(grad, alpha=-lr))


def named_params(model, curr_module=None, memo=None, prefix=''):
    if memo is None:
        memo = set()

    if hasattr(curr_module, 'named_leaves'):
        for name, p in curr_module.named_leaves():
            if p is not None and p not in memo:
                memo.add(p)
                yield prefix + ('.' if prefix else '') + name, p
    else:
        for name, p in curr_module._parameters.items():
            if p is not None and p not in memo:
                memo.add(p)
                yield prefix + ('.' if prefix else '') + name, p

    for mname, module in curr_module.named_children():
        submodule_prefix = prefix + ('.' if prefix else '') + mname
        for name, p in named_params(model, module, memo, submodule_prefix):
            yield name, p

def update_params(model, lr_inner, first_order=False, source_params=None, detach=False):
    '''
        official implementation
    '''
    if source_params is not None:
        for tgt, src in zip(model.named_parameters(), source_params):
            name_t, param_t = tgt
            # name_s, param_s = src
            # grad = param_s.grad
            # name_s, param_s = src
            if src is None:
                print('skip param')
                continue
            #grad = src
            tmp = param_t - lr_inner * src
            set_param(model, model, name_t, tmp)
    return model

def set_param(model, curr_mod, name, param):
    if '.' in name:
        n = name.split('.')
        module_name = n[0]
        rest = '.'.join(n[1:])
        for name, mod in curr_mod.named_children():
            if module_name == name:
                set_param(model, mod, rest, param)
                break
    else:
        setattr(getattr(curr_mod, name), 'data', param)

