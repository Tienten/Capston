import math
import numpy as np
import torchvision.models as models
import torch.utils.data as data
from torchvision import transforms, datasets
import cv2
import torch.nn.functional as F
from torch.autograd import Variable
import os ,torch
import torch.nn as nn
import image_utils
import argparse,random
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kdef', type=str, default='../data_2/KDEF/', help='dataset path.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Pytorch checkpoint file path')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Pretrained weights')
    parser.add_argument('--beta', type=float, default=0.7, help='Ratio of high importance group in one mini-batch.')
    parser.add_argument('--relabel_epoch', type=int, default=1, help='Relabeling samples on each mini-batch after 10(Default) epochs.')
    parser.add_argument('--margin_1', type=float, default=0.15, help='Rank regularization margin. Details described in the paper.')
    parser.add_argument('--margin_2', type=float, default=0.2, help='Relabeling margin. Details described in the paper.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size.')
    parser.add_argument('--optimizer', type=str, default="sgd", help='Optimizer, adam or sgd.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate for sgd.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=70, help='Total training epochs.')
    parser.add_argument('--drop_rate', type=float, default=0, help='Drop out rate.')
    return parser.parse_args()


class KDEFDataset(data.Dataset):
    def __init__(self, root_dir, phase, transform=None, basic_aug=False):
        self.root_dir = root_dir
        self.phase = phase
        self.transform = transform
        self.basic_aug = basic_aug
        self.aug_func = [self.flip_image, self.add_gaussian_noise]
        self.file_paths = []
        self.labels = []
        self.targets = []
        self.label_map = {'AF': 0, 'AN': 1, 'DI': 2, 'HA': 3, 
                          'NE': 4, 'SA': 5, 'SU': 6}

        phase_dir = os.path.join(root_dir, phase)
        print(phase_dir)
        for label in os.listdir(phase_dir):
            label_dir = os.path.join(phase_dir, label)
            print(label_dir)
            if os.path.isdir(label_dir) and (self.phase in label_dir):
                for f in os.listdir(label_dir):
                    if f.endswith('.jpg'):
                        files = os.path.join(label_dir, f)
                        self.file_paths.append(files) 
                        self.targets.append(int(self.label_map[label]))
                        if self.phase == "train":
                            if(f[0:2]=='AF'):
                                self.labels.append(int(self.label_map['AF']))
                            elif(f[0:2]=='AN'):
                                self.labels.append(int(self.label_map['AN']))
                            elif(f[0:2]=='DI'):
                                self.labels.append(int(self.label_map['DI']))
                            elif(f[0:2]=='HA'):
                                self.labels.append(int(self.label_map['HA']))
                            elif(f[0:2]=='NE'):
                                self.labels.append(int(self.label_map['NE']))
                            elif(f[0:2]=='SA'):
                                self.labels.append(int(self.label_map['SA']))
                            elif(f[0:2]=='SU'):
                                self.labels.append(int(self.label_map['SU']))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image_path = self.file_paths[idx]
        image = Image.open(image_path).convert('RGB')
        target = self.targets[idx]
        if self.phase == "train":
            label = self.labels[idx]

        if self.phase == 'train' and self.basic_aug:
            if random.uniform(0, 1) > 0.5:
                index = random.randint(0, 1)
                image = image

        if self.transform:
            image = self.transform(image)

        if self.phase == "train":
            return image, target, label, idx
        else:
            return image, target, idx

    def flip_image(self, image):
        return image.transpose(Image.FLIP_LEFT_RIGHT)

    def add_gaussian_noise(self, image):
        np_image = np.array(image)
        mean = 0
        sigma = 0.1 * 255
        noise = np.random.normal(mean, sigma, np_image.shape)
        noisy_image = np_image + noise
        noisy_image = np.clip(noisy_image, 0, 255)
        return Image.fromarray(noisy_image.astype('uint8'), 'RGB')

class Res18Feature(nn.Module):
    def __init__(self, pretrained = True, num_classes = 7, drop_rate = 0):
        super(Res18Feature, self).__init__()
        self.drop_rate = drop_rate
        resnet  = models.resnet18(pretrained=True)
        # self.feature = nn.Sequential(*list(resnet.children())[:-2]) # before avgpool
        self.features = nn.Sequential(*list(resnet.children())[:-1]) # after avgpool 512x1

        fc_in_dim = list(resnet.children())[-1].in_features # original fc layer's in dimention 512
   
        self.fc = nn.Linear(fc_in_dim, num_classes) # new fc layer 512x7
        self.alpha = nn.Sequential(nn.Linear(fc_in_dim, 1),nn.Sigmoid())

    def forward(self, x):
        x = self.features(x)
        
        if self.drop_rate > 0:
            x =  nn.Dropout(self.drop_rate)(x)
        x = x.view(x.size(0), -1)
        
        attention_weights = self.alpha(x)
        out = attention_weights * self.fc(x)
        return attention_weights, out
        
def initialize_weight_goog(m, n=''):
    # weight init as per Tensorflow Official impl
    # https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_model.py
    # if isinstance(m, CondConv2d):
        # fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        # init_weight_fn = get_condconv_initializer(
            # lambda w: w.data.normal_(0, math.sqrt(2.0 / fan_out)), m.num_experts, m.weight_shape)
        # init_weight_fn(m.weight)
        # if m.bias is not None:
            # m.bias.data.zero_()
    if isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        fan_out = m.weight.size(0)  # fan-out
        fan_in = 0
        if 'routing_fn' in n:
            fan_in = m.weight.size(1)
        init_range = 1.0 / math.sqrt(fan_in + fan_out)
        m.weight.data.uniform_(-init_range, init_range)
        m.bias.data.zero_()
        
def run_training():
    args = parse_args()
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(scale=(0.02,0.25))
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
    } 
    
    train_dataset = KDEFDataset(args.kdef, phase = 'train', transform = data_transforms['train'], basic_aug = True)
    print('Train set size:', train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = True,  
                                               pin_memory = True)
    
    device = torch.device("cpu")
    
    training_acc=[]
    
    imagenet_pretrained = True
    res18 = Res18Feature(pretrained = imagenet_pretrained, drop_rate = args.drop_rate) 
    if not imagenet_pretrained:
         for m in res18.modules():
            initialize_weight_goog(m)
            
    if args.pretrained:
        print("Loading pretrained weights...", args.pretrained) 
        pretrained = torch.load(args.pretrained)
        pretrained_state_dict = pretrained['state_dict']
        model_state_dict = res18.state_dict()
        loaded_keys = 0
        total_keys = 0
        for key in pretrained_state_dict:
            if  ((key=='module.fc.weight')|(key=='module.fc.bias')):
                pass
            else:    
                model_state_dict[key] = pretrained_state_dict[key]
                total_keys+=1
                if key in model_state_dict:
                    loaded_keys+=1
        print("Loaded params num:", loaded_keys)
        print("Total params num:", total_keys)
        res18.load_state_dict(model_state_dict, strict = False) 
    
    params = res18.parameters()
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params,weight_decay = 1e-4)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, args.lr,
                                    momentum=args.momentum,
                                    weight_decay = 1e-4)
    else:
        raise ValueError("Optimizer not supported.")
    
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
    res18 = res18.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    
    margin_1 = args.margin_1
    margin_2 = args.margin_2
    beta = args.beta
    best_acc = 0.0
    
    for i in range(1, args.epochs + 1):
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        res18.train()
        for batch_i, (imgs, targets, labels, indexes) in enumerate(train_loader):
            batch_sz = imgs.size(0) 
            iter_cnt += 1
            tops = int(batch_sz* beta)
            optimizer.zero_grad()
            imgs = imgs.to(device)
            attention_weights, outputs = res18(imgs)
            
            # Rank Regularization
            _, top_idx = torch.topk(attention_weights.squeeze(), tops)
            _, down_idx = torch.topk(attention_weights.squeeze(), batch_sz - tops, largest = False)

            high_group = attention_weights[top_idx]
            low_group = attention_weights[down_idx]
            high_mean = torch.mean(high_group)
            low_mean = torch.mean(low_group)
            diff  = low_mean - high_mean + margin_1

            if diff > 0:
                RR_loss = diff
            else:
                RR_loss = 0.0
            
            targets = targets.to(device)
            loss = criterion(outputs, targets) + RR_loss 
            loss.backward()
            optimizer.step()
            
            running_loss += loss
            _, predicts = torch.max(outputs, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num

            # Relabel samples
            if i >= args.relabel_epoch:
                sm = torch.softmax(outputs, dim = 1)
                Pmax, predicted_labels = torch.max(sm, 1) # predictions
                Pgt = torch.gather(sm, 1, targets.view(-1,1)).squeeze() # retrieve predicted probabilities of targets
                true_or_false = Pmax - Pgt > margin_2
                update_idx = true_or_false.nonzero().squeeze() # get samples' index in this mini-batch where (Pmax - Pgt > margin_2)
                label_idx = indexes[update_idx] # get samples' index in train_loader
                relabels = predicted_labels[update_idx] # predictions where (Pmax - Pgt > margin_2)
                train_loader.dataset.labels = np.array(train_loader.dataset.labels)
                train_loader.dataset.labels[label_idx.cpu().numpy()] = relabels.cpu().numpy() # relabel samples in train_loader
                
        scheduler.step()
        acc = correct_sum.float() / float(train_dataset.__len__())
        running_loss = running_loss/iter_cnt
        training_acc.append(float('%.4f'%(acc)))
        print('[Epoch %d] Accuracy: %.4f. Loss: %.3f' % (i, acc, running_loss))
        
        if acc > best_acc:
            best_acc = acc
    print('Best Acc: {:4f}'.format(best_acc))
    print('Accuracy: ', training_acc)

     
            
if __name__ == "__main__": 
    parse_args()      
    run_training()