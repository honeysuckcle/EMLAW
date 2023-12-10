from __future__ import print_function
import yaml
import easydict
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from apex import amp, optimizers
from utils.utils import log_set, save_model
from utils.loss import ova_loss, open_entropy, consistency_loss
from utils.lr_schedule import inv_lr_scheduler
from utils.defaults import get_dataloaders, get_models
from eval import test
import argparse
from utils.LogitNorm import LogitNorm

parser = argparse.ArgumentParser(description='Pytorch OVANet',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config', type=str, default='configs/office-train-config_OPDA.yaml',
                    help='/path/to/config/file')

parser.add_argument('--source_data', type=str,
                    default='./txt/source_amazon_opda.txt',
                    help='path to source list')
parser.add_argument('--target_data', type=str,
                    default='./txt/target_dslr_opda.txt',
                    help='path to target list')
parser.add_argument('--log-interval', type=int,
                    default=100,
                    help='how many batches before logging training status')
parser.add_argument('--exp_name', type=str,
                    default='office',
                    help='/path/to/config/file')
parser.add_argument('--network', type=str,
                    default='resnet50',
                    help='network name')
parser.add_argument("--gpu_devices", type=int, nargs='+',
                    default=[0], help="")
parser.add_argument("--no_adapt",
                    default=False, action='store_true')
parser.add_argument("--save_model",
                    default=False, action='store_true')
parser.add_argument("--save_path", type=str,
                    default="record/ova_model",
                    help='/path/to/save/model')
parser.add_argument('--multi', type=float,
                    default=0.1,
                    help='weight factor for adaptation')
parser.add_argument('--cutoff', type=float,
                    default=0.1,
                    help='weight factor for adaptation')
parser.add_argument('--lambda_o', type=float,
                    default=0.1,
                    help='weight factor for outlier classification')
parser.add_argument('--lambda_fix', type=float,
                    default=0.1,
                    help='weight factor for outlier classification')
args = parser.parse_args()

config_file = args.config
conf = yaml.load(open(config_file))
save_config = yaml.load(open(config_file))
conf = easydict.EasyDict(conf)
gpu_devices = ','.join([str(id) for id in args.gpu_devices])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
args.cuda = torch.cuda.is_available()

cutoff = args.cutoff
lambda_o = args.lambda_o
lambda_fix = args.lambda_fix
source_data = args.source_data
target_data = args.target_data
evaluation_data = args.target_data
network = args.network
use_gpu = torch.cuda.is_available()
n_share = conf.data.dataset.n_share
n_source_private = conf.data.dataset.n_source_private
n_total = conf.data.dataset.n_total
open = n_total - n_share - n_source_private > 0
num_class = n_share + n_source_private
script_name = os.path.basename(__file__)

inputs = vars(args)
inputs["evaluation_data"] = evaluation_data
inputs["conf"] = conf
inputs["script_name"] = script_name
inputs["num_class"] = num_class
inputs["config_file"] = config_file

source_loader, target_loader, \
test_loader, target_folder = get_dataloaders(inputs)

logname = log_set(inputs)

G, C1, C2, O, opt_g, opt_c, \
param_lr_g, param_lr_c = get_models(inputs)
ndata = target_folder.__len__()


def train():
    criterion = nn.CrossEntropyLoss().cuda()
    print('train start!')
    data_iter_s = iter(source_loader)
    data_iter_t = iter(target_loader)
    len_train_source = len(source_loader)
    len_train_target = len(target_loader)
    for step in range(conf.train.min_step + 1):
        G.train()
        C1.train()
        C2.train()
        O.train()
        if step % len_train_target == 0:
            data_iter_t = iter(target_loader)
        if step % len_train_source == 0:
            data_iter_s = iter(source_loader)
        data_t = next(data_iter_t)
        data_s = next(data_iter_s)
        inv_lr_scheduler(param_lr_g, opt_g, step,
                         init_lr=conf.train.lr,
                         max_iter=conf.train.min_step)
        inv_lr_scheduler(param_lr_c, opt_c, step,
                         init_lr=conf.train.lr,
                         max_iter=conf.train.min_step)
        img_s = data_s['x']
        label_s = data_s['y']
        img_t, img_t_s = data_t['x_w'], data_t['x_s']
        img_s, label_s = Variable(img_s.cuda()), \
                         Variable(label_s.cuda())
        img_t, img_t_s = Variable(img_t.cuda()), Variable(img_t_s.cuda())
        opt_g.zero_grad()
        opt_c.zero_grad()
        C2.module.weight_norm()

        ## Source loss calculation
        feat = G(img_s)
        out_s = C1(feat)
        out_open = C2(feat)
        ## source classification loss
        loss_s = criterion(out_s, label_s)
        ## open set loss for source
        out_open = out_open.view(out_s.size(0), 2, -1)
        open_loss_pos, open_loss_neg = ova_loss(out_open, label_s)
        ## b x 2 x C
        loss_open = 0.5 * (open_loss_pos + open_loss_neg)
        ## open set loss for target
        all = loss_s + loss_open
        log_string = 'Train {}/{} \t ' \
                     'Loss Source: {:.4f} ' \
                     'Loss Open: {:.4f} ' \
                     'Loss Open Source Positive: {:.4f} ' \
                     'Loss Open Source Negative: {:.4f} '
        log_values = [step, conf.train.min_step,
                      loss_s.item(),  loss_open.item(),
                      open_loss_pos.item(), open_loss_neg.item()]
        if not args.no_adapt:
            feat_t = G(img_t)
            out_open_t = C2(feat_t)
            out_open_t = out_open_t.view(img_t.size(0), 2, -1)
            ent_open = open_entropy(out_open_t)
            all += args.multi * ent_open
            log_values.append(ent_open.item())
            log_string += "Loss Open Target: {:.6f} "
            with torch.no_grad():
                logit_c = C1(feat_t)
                logit_c = F.softmax(logit_c, 1)
                logit_mb = out_open_t.detach()
                logit_mb = F.softmax(logit_mb, 1)
                tmp_range = torch.arange(0, logit_mb.size(0)).long().cuda()
                o_neg = logit_mb[tmp_range, 0, :]
                Si = torch.sum(logit_c * o_neg, 1)
                in_mask = (Si < cutoff)
                out_mask = (Si > (1-cutoff))
            in_x = feat_t[in_mask]
            num_in = len(in_x)
            out_x = feat_t[out_mask]
            num_out = len(out_x)
            target = torch.cat((torch.zeros(num_in), torch.ones(num_out)), 0).long().cuda()
            logit_out = O(torch.cat((in_x, out_x), 0))
            out_loss = criterion(logit_out, target) + 1e-8
            all += lambda_o * out_loss
            log_values.append(out_loss.item())
            log_string += "Classification Loss Outlier Classifier: {:.6f} "
            # FixMatch
            outlier_inputs = torch.cat((img_t, img_t_s), 0)
            logit_outlier_w, logit_outlier_s = O(G(outlier_inputs)).chunk(2)
            logit_outlier_w, logit_outlier_s = F.softmax(logit_outlier_w, 1), F.softmax(logit_outlier_s, 1)
            fix_loss = consistency_loss(logit_outlier_w, logit_outlier_s, 'ce')
            all += lambda_fix * fix_loss
            log_values.append(fix_loss.item())
            log_string += "Consistency Loss Outlier Classifier: {:.6f} "


        with amp.scale_loss(all, [opt_g, opt_c]) as scaled_loss:
            scaled_loss.backward()
        opt_g.step()
        opt_c.step()
        opt_g.zero_grad()
        opt_c.zero_grad()
        if step % conf.train.log_interval == 0:
            print(log_string.format(*log_values))
        if step > 0 and step % conf.test.test_interval == 0:
            acc_o, h_score = test(step, test_loader, logname, n_share, G,
                                  [C1, O], open=open)
            print("acc all %s h_score %s " % (acc_o, h_score))
            G.train()
            C1.train()
            C2.train()
            O.train()
            if args.save_model:
                save_path = "%s_%s.pth"%(args.save_path, step)
                save_model(G, C1, C2, O, save_path)


train()
