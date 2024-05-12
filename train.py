import os
import math
import tempfile
import argparse
from yacs.config import CfgNode as CN

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_dataset import MyDataSet
from utils import read_split_data_mode, plot_data_loader_image
from multi_train_utils.distributed_utils import init_distributed_mode, dist, cleanup
from multi_train_utils.train_eval_utils import train_one_epoch, evaluate

'''
from mlp_mixer_pytorch.mlp_mixer_pytorch import MLPMixer
from mlp_mixer_pytorch.mlp_mixer_pytorch import Transpose
from mlp_mixer_pytorch.mlp_mixer_pytorch import SegBlock
from mlp_mixer_pytorch.mlp_mixer_pytorch import Segmentation
'''

from mlp_mixer_pytorch import MLPMixer

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
yaml_name = 'configs/kanmixer.yaml'  # 参数配置文件
fcfg = open(yaml_name)
cfg = CN.load_cfg(fcfg)

def main(args):
    # 检查是否有可用的GPU设备
    if torch.cuda.is_available() is False:
        # 如果没有，抛出环境错误
        raise EnvironmentError("not find GPU device for training.")

    # 初始化分布式环境
    init_distributed_mode(args=args)

    # 获取当前进程的rank
    rank = args.rank
    # 获取设备类型
    device = torch.device(args.device)
    # 获取批处理大小
    batch_size = args.batch_size
    # 获取权重路径
    weights_path = args.weights
    # 根据并行GPU的数量调整学习率
    args.lr *= args.world_size
    # 初始化检查点路径
    checkpoint_path = ""

    # 如果是第一个进程
    if rank == 0:
        # 打印参数信息
        print(args)
        # 打印Tensorboard启动信息
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        # 实例化Tensorboard的写入器
        tb_writer = SummaryWriter()
        # 如果权重目录不存在，则创建
        if os.path.exists("./weights") is False:
            os.makedirs("./weights")

    train_images_path, train_images_label = read_split_data_mode(args.data_path, mode="train", supported=[".JPEG"])
    val_images_path, val_images_label = read_split_data_mode(args.data_path, mode="val", supported=['.JPEG'])

    num_classes = 1000
    # check num_classes
    assert args.num_classes == num_classes, "dataset num_classes: {}, input {}".format(args.num_classes,
                                                                                       num_classes)

    img_size = 224
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_data_set = MyDataSet(images_path=train_images_path,
                               images_class=train_images_label,
                               transform=data_transform["train"])

    # 实例化验证数据集
    val_data_set = MyDataSet(images_path=val_images_path,
                             images_class=val_images_label,
                             transform=data_transform["val"])

    # 给每个rank对应的进程分配训练的样本索引
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data_set)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data_set)

    # 将样本索引每batch_size个元素组成一个list
    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, batch_size, drop_last=True)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    if rank == 0:
        print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_data_set.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_data_set.collate_fn)
    
    
    # 构建MLP-Mixer模型
    model = MLPMixer(
        image_size = 224,  # 图像的高和宽
        channels = 3,  # 图像的通道数
        patch_size = 32,  # MLP-Mixer的patch大小
        dim = 512,  # MLP-Mixer的维度
        depth = 8,  # MLP-Mixer的深度
        num_classes = 1000)  # 输出类别数量

    # 如果存在预训练权重则载入
    if os.path.exists(weights_path):
        weights_dict = torch.load(weights_path, map_location=device)
        load_weights_dict = {k: v for k, v in weights_dict.items()
                             if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(load_weights_dict, strict=False)
    else:
        checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
        # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
        if rank == 0:
            torch.save(model.state_dict(), checkpoint_path)

        dist.barrier()
        # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            if "fc" not in name:
                para.requires_grad_(False)
    else:
        # 只有训练带有BN结构的网络时使用SyncBatchNorm采用意义
        if args.syncBN:
            # 使用SyncBatchNorm后训练会更耗时
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    # 转为DDP模型
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

    # optimizer
    pg = [p for p in model.parameters() if p.requires_grad]

    #optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=0.005)
    optimizer = optim.AdamW(pg, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.1)

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)

        mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)

        scheduler.step()

        sum_num = evaluate(model=model,
                           data_loader=val_loader,
                           device=device)
        acc = sum_num / val_sampler.total_size

        if rank == 0:
            print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
            tags = ["loss", "accuracy", "learning_rate"]
            tb_writer.add_scalar(tags[0], mean_loss, epoch)
            tb_writer.add_scalar(tags[1], acc, epoch)
            tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

            torch.save(model.module.state_dict(), "./weights/model-{}.pth".format(epoch))

    # 删除临时缓存文件
    if rank == 0:
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)

    cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=768)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.1)
    # 是否启用SyncBatchNorm
    parser.add_argument('--syncBN', type=bool, default=True)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str, default="/mnt/zhaojiabao-23/MLP-MIXER-PYTORCH-新/ILSVRC_train_t12")

    # resnet34 官方权重下载地址
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    parser.add_argument('--weights', type=str, default='/mnt/zhaojiabao-23/MLP-MIXER-PYTORCH-新/FFKAN_MLP_IMAGENET_1K_03/weights/model-0.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    # 不要改该参数，系统会自动分配
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    opt = parser.parse_args()

    main(opt)