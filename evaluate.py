import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.dice_score import multiclass_dice_coeff, dice_coeff
from torch import Tensor


def evaluate(net,dataloader,device,metrics):
    metrics.reset
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    test_accuracy = 0
    val_iou_score = 0

    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        targets = mask_true.cpu().numpy()
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)
            preds = mask_pred.detach().max(dim=1)[1].cpu().numpy()
            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # preds= mask_pred.detach.cpu().numpy()
                metrics.update(targets, preds)
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)


           

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    score = metrics.get_results()    
    return dice_score / num_val_batches,score


   