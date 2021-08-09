import os
import argparse
import csv
import sys
sys.path.append('./apex')
try:
    import apex
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

import torch
from data.my_loader_pix import load_data as load_pix
from data.my_loader_psc import load_data as load_psc
from model.svda import SVDA
from model.sada import SADA
from model.sida import SIDA

from test import test

def main(args):
    # environment setting
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    # logging file
    with open(os.path.join(args.checkpoint_dir, 'train_valid.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['Iter', 'train_loss', 'valid_loss', 'valid_iou', 'best_iter', 'best_iou'])

    # construct model
    if args.test_data == 'pixel':
        train_loader, test_loader = load_pix(args)
    else:
        train_loader, test_loader = load_psc(args)

    if args.model_type == 'v':
        from train import train
        mdl = SVDA(pretrain_path=args.pretrain_path, naive=args.naive).cuda()
    elif args.model_type == 'a':
        from train import train
        mdl = SADA(pretrain_path=args.pretrain_path, naive=args.naive).cuda()
    elif args.model_type == 'i':
        from train_info import train
        mdl = SIDA(args.pretrain_path, args).cuda()
    else:
        print('No such type of model')

    if args.resume_path:
        print('Resume from', args.resume_path)
        mdl.load_state_dict(torch.load(os.path.join(args.data_dir, 
                                        'checkpoint', args.resume_path))) 
     
    if args.sync_bn:
        mdl = apex.parallel.convert_syncbn_model(mdl)
    mdl = torch.nn.parallel.DistributedDataParallel(mdl, device_ids=[args.local_rank],
                                                    output_device=args.local_rank)
    # mdl = DDP(mdl, delay_allreduce=True)
    mdl = mdl.module
    for param in mdl.encoder.parameters():
        param.requires_grad = False
    for param in mdl.decoder.parameters():
        param.requires_grad = False

    if args.eval: # eval only
        loss, iou = test(mdl, test_loader, device)
        print('Exp %s Test result: loss %4f iou %4f' % (args.exp_name, loss, iou))
    else:
        train(mdl, train_loader, test_loader, device, args)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument("--epoch_num", action="store", dest="epoch_num", default=100, type=int, help="Epoch to train [200]")
    parser.add_argument("--learning_rate", action="store", dest="lr", default=4e-5, type=float, help="Learning rate for adam ")
    parser.add_argument("--batch_size", action="store", dest="batch_size", default=32, type=int, help="batch_size for training [32]")
    parser.add_argument("--workers", action="store", dest="workers", default=0, type=int, help="worker num for dataloader")
    parser.add_argument("--vox_size", action="store", dest="vox_size", default=32, type=int, help="Voxel resolution for training [32]")
    parser.add_argument("--data_dir", action="store", dest="data_dir", default='/mnt4/fqy/dataset/3D', help="Root directory of dataset [dbs]")
    parser.add_argument("--checkpoint_dir", action="store", dest="checkpoint_dir", default="checkpoint", help="Directory name to save the checkpoints [trained]")
    parser.add_argument("--exp_name", action="store", dest="exp_name", default="svda")
    parser.add_argument("--pretrain_path", action="store", default="vxl_naive_ae-49.pth", help="pretrain voxel model path")
    # tuning
    parser.add_argument("--lamb_d", action="store", dest="lamb_d", default=0.001, type=float, help="loss balance term")
    parser.add_argument("--lamb_s", action="store", dest="lamb_s", default=0.0001, type=float, help="loss balance term")
    parser.add_argument("--lamb_a", action="store", dest="lamb_a", default=0.01, type=float, help="loss balance term")
    parser.add_argument("--lamb_b", action="store", dest="lamb_b", default=0.01, type=float, help="loss balance term")
    parser.add_argument("--lamb_g", action="store", dest="lamb_g", default=0.01, type=float, help="loss balance term")
    parser.add_argument("--implicit", action="store_true", dest="implicit", default=False, help="apply implicit model as 3D backbone")
    parser.add_argument("--eval", action="store_true", dest="eval", default=False, help="True for testing only")
    parser.add_argument("--resume_path", action="store", default="", help="resume model path")
    parser.add_argument("--test_data", action="store", default="pixel", help="[pixel3d, pascal3d]")
    # distribution
    parser.add_argument('--fp16', action='store_true', 
                        help='use float16 instead of float32, which will save about 50% memory' )
    parser.add_argument("--distributed", action="store_true", default=False, 
                        help="multi gpu device.")
    parser.add_argument("--multiprocessing_distributed", action="store_true", default=False, 
                        help="multi gpu device.")
    parser.add_argument('--world_size', type=int, default=1, 
                        help='total number of gpus' )
    parser.add_argument('--master-port', default='9901', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--master-addr', default='127.0.0.1', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:9876', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--sync_bn', action='store_true',
                        help='enabling apex sync BN.')
    parser.add_argument("--local_rank", default=0, type=int,
                        help='must parsing from the command')
    args = parser.parse_args()

    print('Using multiple gpus: %d' % torch.cuda.device_count())
    os.environ['MASTER_ADDR'] = args.master_addr #'127.0.0.1'
    os.environ['MASTER_PORT'] = '9903' #args.master_port #

    args.gpu = args.local_rank
    torch.cuda.set_device(args.gpu)
    torch.distributed.init_process_group(backend='nccl',
                                            init_method='env://')
    world_size = torch.distributed.get_world_size()
    
    #  --data_dir '/home/qianyu/fqy/data/3D'

    # add root to dirs
    args.checkpoint_dir = os.path.join(args.data_dir, 'checkpoint')

    # implicit model
    if args.implicit:
        args.pretrain_path = 'vxl_ae_best.pth'
    args.pretrain_path = os.path.join(args.checkpoint_dir, args.pretrain_path)
    
    print('\n=== Model type %s ===\n' % args.model_type)

    # create checkpoint dir
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.exp_name)
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    main(args)
