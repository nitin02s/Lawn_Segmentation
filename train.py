import argparse
import logging
import sys
from pathlib import Path
import torchvision.transforms as T
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from utils.accuracy import mIoU,pixel_accuracy
from utils.data_loading import BasicDataset
from utils.dice_score import dice_loss
from evaluate import evaluate
from unet import UNet
from utils.metrics import StreamSegMetrics
from utils.loss import FocalLoss

# image directory
dir_img = Path('/home/somashekhar.n/project/test_images/imgs') 

#masked directory
dir_mask = Path('/home/somashekhar.n/project/test_images/masks')

#directory to save checkpoint
dir_checkpoint = Path('/home/somashekhar.n/project/Pytorch-UNet/weights_plant/')

def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 32,
              learning_rate: float = 1e-5,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False):
    # 1. Create dataset
    
    dataset = BasicDataset(dir_img, dir_mask, img_scale)    

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=1, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')


    metric = StreamSegMetrics(net.n_classes)

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-7, momentum=0.9)

    #adam optimizer
    # optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-4)
    # scheduler 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, verbose=True)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # Cross entropy loss
    # criterion = nn.CrossEntropyLoss()

    #focal loss
    criterion = FocalLoss(ignore_index=255, size_average=True)
    global_step = 0
    transform = T.ToPILImage()
    # 5. Begin training
    for epoch in range(1, epochs+1):
        net.train()
        epoch_loss = 0
        val_score_l = []
        print("classes {}".format(net.n_classes))
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:

                # i=i+1
                # print(i)
                images = batch['image']
                true_masks = batch['mask']
                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast():
                    masks_pred = net(images)
                    loss = criterion(ignore_index=255, size_average=True)
                optimizer.zero_grad()
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                print({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (10 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in net.named_parameters():
                            tag = tag.replace('/', '.')
                            if not torch.isinf(value).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not torch.isinf(value.grad).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score,score = evaluate(net,val_loader,device,metric)
                        print(metric.to_str(score))
                        val_score_l.append(val_score)
                        scheduler.step(val_score)
                        pbar.set_postfix(**{'validation score (batch)': val_score})

                        logging.info('Validation Dice score: {}'.format(val_score))
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation Dice': val_score,
                            'images': wandb.Image(images[0].cpu()),
                            'masks': {
                                'true': wandb.Image(transform(true_masks[0].float().cpu())),
                                'pred': wandb.Image(transform(masks_pred.argmax(dim=1)[0].float().cpu())),
                            },
                            'step': global_step,
                            'epoch': epoch,
                            'train_loss':loss.item(),
                            'Overall Acc':score['Overall Acc'],
                            'Mean Acc':score["Mean Acc"],
                            'FrewW Acc':score["FreqW Acc"],
                            "Mean IOU":score["Mean IoU"],
                            "Class IoU":score["Class IoU"],
                            **histograms
                        })

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':


    # Hyperparameters to set for training

    classes=3
    bilinear= False
    epochs = 10
    batch_size = 8
    lr = 1e-3
    scale = 0.5
    val = 20.0
    amp = False

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    #loading model
    net = UNet(n_channels=3, n_classes=classes, bilinear=bilinear)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')


    #pre loaded model
    # if args.load:
    # net.load_state_dict(torch.load("/home/somashekhar.n/project/Pytorch-UNet/weights/checkpoint_epoch5.pth", map_location=device))
    # print('Model pre loaded')

    net.to(device=device)
    try:
        train_net(net=net,
                  epochs=epochs,
                  batch_size=batch_size,
                  learning_rate=lr,
                  device=device,
                  img_scale=scale,
                  val_percent=val / 100,
                  amp=amp)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise
