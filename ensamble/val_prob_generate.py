import os
import argparse
import pandas as pd
import numpy as np

from nn_models.backbone import ProductNet
#from nn_models.SeResnet152 import ProductNet
#from nn_models.InceptionResnetV2 import ProductNet
#from nn_models.NasNet import ProductNet
from utils.data import ProductDataset
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import transforms
import pickle
import pretrainedmodels
import pretrainedmodels.utils
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
model_names = sorted(name for name in pretrainedmodels.__dict__
                     if not name.startswith("__")
                     and name.islower()
                     and callable(pretrainedmodels.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Product Training')
parser.add_argument('--pretrained', default='imagenet', help='use pre-trained model')
parser.add_argument('--arch', '-a', metavar='ARCH', default='se_resnet50', #nasnetamobile
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: fbresnet152)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-b', '--batch_size', default=24, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--output', default=os.path.join('result', 'warmup_autoaug'))

INPUT_DIR = os.path.join(os.getcwd(), 'input')

Normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#Normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
test_augment = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.TenCrop((224,224)),
    transforms.Lambda(lambda crops: torch.stack([Normalize(transforms.ToTensor()(crop)) for crop in crops])),
    ])
#test_augment = transforms.Compose([
#    transforms.Resize((320,320)),
#    transforms.TenCrop((299,299)),
#    transforms.Lambda(lambda crops: torch.stack([Normalize(transforms.ToTensor()(crop)) for crop in crops])),
#    ])
#test_augment = transforms.Compose([
#    transforms.Resize((350,350)),
#    transforms.TenCrop((331,331)),
#    transforms.Lambda(lambda crops: torch.stack([Normalize(transforms.ToTensor()(crop)) for crop in crops])),
#    ])


def predict():
    args = parser.parse_args()
    print(args)
    args.resume = os.path.join(args.output,'checkpoint', args.resume)

    test_dataset = ProductDataset(test_augment, 'val.txt', 'val')
    test_loader = DataLoader(
                    test_dataset,
                    sampler     = SequentialSampler(test_dataset),
                    batch_size  = args.batch_size,
                    drop_last   = False,
                    num_workers = 16,
                    pin_memory  = True)

    # create model
    model = ProductNet(args.arch, args.pretrained)
    model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            prec3 = checkpoint['prec3']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    model.eval()

    all_id = []
    all_pred = []
    all_probs = []
    all_num = 0
    with torch.no_grad():
        for i, (image_id, input, target) in enumerate(tqdm(test_loader)):
            bs, ncrops, c, h, w = input.size()
            input = input.cuda()
            # compute output
            output = model(input.view(-1, c, h, w))
            output = output.view(bs, ncrops, -1).mean(1)
            _, pred = output.topk(3, 1, True, True)
            pred = pred.squeeze().data.cpu().numpy()
            all_probs.append(F.log_softmax(output, dim=1))
            all_id.append(image_id)
            all_pred.append(pred)
            all_num += len(image_id)
    assert(all_num == len(test_loader.sampler))
    all_id = np.concatenate(all_id)
    all_pred = np.concatenate(all_pred).astype(int)
    all_probs = np.concatenate(all_probs)
    print(all_id.shape, all_pred.shape)
    prob_file = os.path.join(args.output, 'val_'+args.output.split('/')[-1] + '_epoch_' + str(checkpoint['epoch']) +'_prec3_'+ str(prec3) +'_tta.pkl')
    data = {'id': all_id, 'probs':all_probs}
    f = open(prob_file, 'wb')
    pickle.dump(data, f)
    f.close()


if __name__ == '__main__':
    predict()








