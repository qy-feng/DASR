import argparse


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument("--epoch_num", action="store", dest="epoch_num", default=40, type=int, help="Epoch to train [200]")
    parser.add_argument("--learning_rate", action="store", dest="lr", default=4e-5, type=float, help="Learning rate for adam ")
    parser.add_argument("--batch_size", action="store", dest="batch_size", default=32, type=int, help="batch_size for training [32]")
    parser.add_argument("--workers", action="store", dest="workers", default=0, type=int, help="worker num for dataloader")
    parser.add_argument("--lamb_d", action="store", dest="lamb_d", default=0.001, type=float, help="loss balance term")
    parser.add_argument("--lamb_s", action="store", dest="lamb_s", default=0.0001, type=float, help="loss balance term")
    parser.add_argument("--vox_size", action="store", dest="vox_size", default=32, type=int, help="Voxel resolution for training [32]")

    parser.add_argument("--data_dir", action="store", dest="data_dir", default="/data/qianyu/3d/data", help="Root directory of dataset [dbs]")
    parser.add_argument("--checkpoint_dir", action="store", dest="checkpoint_dir", default="checkpoints", help="Directory name to save the checkpoints [trained]")
    parser.add_argument("--exp_name", action="store", dest="exp_name", default="svda")
    parser.add_argument("--pretrain_path", action="store", default="vxl_naive_ae-199.pth", help="pretrain voxel model path")

    parser.add_argument("--implicit", action="store_true", dest="implicit", default=False, help="apply implicit model as 3D backbone")
    parser.add_argument("--baseline", action="store_true", dest="baseline", default=False, help="baseline method")
    parser.add_argument("--lamb_a", action="store", dest="lamb_a", default=0.01, type=float, help="loss balance term")
    parser.add_argument("--lamb_b", action="store", dest="lamb_b", default=0.01, type=float, help="loss balance term")
    parser.add_argument("--lamb_g", action="store", dest="lamb_g", default=0.01, type=float, help="loss balance term")
    parser.add_argument("--eval", action="store_true", dest="eval", default=False, help="True for testing only")
    parser.add_argument("--resume_path", action="store", default="", help="resume model path")
    parser.add_argument("--test_data", action="store", default="pixel", help="pixel, pascal")
    parser.add_argument("--distributed", action="store_true", default=False, 
                        help="multi gpu device.")

    args = parser.parse_args()

    return args
