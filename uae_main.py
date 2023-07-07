import models
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import xray_data
import matplotlib.pyplot as plt
import random
from sklearn import metrics, neighbors, mixture, svm
from sklearn import decomposition, manifold
from tqdm import tqdm
from argparse import ArgumentParser
from tensorboardX import SummaryWriter
from torchvision.utils import save_image


WORKERS = 4
torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
np.random.seed(0)

def train(opt):
    # Sets the device to cuda
    device = torch.device('cuda:{}'.format(opt.cuda))
    torch.cuda.set_device('cuda:{}'.format(opt.cuda))
    opt.device = device
    
    model = models.AE(opt.ls, opt.mp, opt.u, img_size=opt.image_size)
    model.to(device)

    EPOCHS = opt.epochs
    # Loads the data, loader is the training data, test_loader is the testing data
    loader = xray_data.get_xray_dataloader(
        opt.batchsize, WORKERS, 'train', img_size=opt.image_size, dataset=opt.dataset)
    test_loader = xray_data.get_xray_dataloader(
        opt.batchsize, WORKERS, 'test', img_size=opt.image_size, dataset=opt.dataset)

    opt.epochs = EPOCHS
    train_loop(model, loader, test_loader, opt)

def train_loop(model, loader, test_loader, opt):
    device = torch.device('cuda:{}'.format(opt.cuda))
    print(opt.exp)
    optim = torch.optim.Adam(model.parameters(), 5e-4, betas=(0.5, 0.999))
    writer = SummaryWriter('log/%s' % opt.exp)
    # Trains the data for each epoch
    for e in tqdm(range(opt.epochs)):
        l1s, l2s = [], []
        model.train()
        # Trains the data for each batch
        for (x, _) in tqdm(loader):
            x = x.to(device)
            x.requires_grad = False
            # Uses the VAE to reconstruct the data
            if not opt.u:
                out = model(x)
                rec_err = (out - x) ** 2
                loss = rec_err.mean()
                l1s.append(loss.item())
            # Uses the UPAE to reconstruct the data
            else:
                # mean is the reconstructed data, logvar is the variance
                mean, logvar = model(x)
                # reconstruction error
                rec_err = (mean - x) ** 2
                loss1 = torch.mean(torch.exp(-logvar)*rec_err)
                loss2 = torch.mean(logvar)
                loss = loss1 + loss2
                l1s.append(rec_err.mean().item())
                l2s.append(loss2.item())

            # Backpropagation
            optim.zero_grad()
            loss.backward()
            optim.step()
        # Tests the data for each epoch
        auc = test_for_xray(opt, model, test_loader, writer=writer, epoch=e)
        # Saves the data in the tensorboard
        # VAE
        if not opt.u:
            # Saves the AUC, Reconstruction Error, and Reconstructions of the last batch per epoch
            l1s = np.mean(l1s)
            writer.add_scalar('auc', auc, e)
            writer.add_scalar('rec_err', l1s, e)
            writer.add_images('recons', torch.cat((x, out)).cpu()*0.5+0.5, e)
            print('epochs:{}, recon error:{}'.format(e, l1s))
        else:
            # Saves the AUC, Reconstruction Error, Variance, Reconstructions, and Variances of the last batch per epoch
            l1s = np.mean(l1s)
            l2s = np.mean(l2s)
            writer.add_scalar('auc', auc, e)
            writer.add_scalar('rec_err', l1s, e)
            writer.add_scalar('logvars', l2s, e)
            writer.add_images('recons', torch.cat((x, mean)).cpu()*0.5+0.5, e)
            writer.add_images('vars', torch.cat(
                (x*0.5+0.5, logvar.exp())).cpu(), e)
            print('epochs:{}, recon error:{}, logvars:{}'.format(e, l1s, l2s))

    # Saves the model
    torch.save(model.state_dict(),
               './models/{}.pth'.format(opt.exp))


# Tests the model
def test_for_xray(opt, model=None, loader=None, plot=False, vae=False, plot_name="test", writer = None, epoch = None):
    # Loads the model and data
    if model is None:
        device = torch.device('cuda:{}'.format(opt.cuda))
        torch.cuda.set_device('cuda:{}'.format(opt.cuda))
        opt.device = device
        model = models.AE(opt.ls, opt.mp, opt.u,
                                img_size=opt.image_size, vae=vae).to(opt.device)
        model.load_state_dict(torch.load(
            './models/{}.pth'.format(opt.exp)))
    if loader is None:
        loader = xray_data.get_xray_dataloader(
            1, WORKERS, 'test', dataset=opt.dataset, img_size=opt.image_size)

    model.eval()
    with torch.no_grad():
        # Store the abnormality scores and labels
        y_score, y_true = [], []
        # Tests the data for each batch
        for bid, (x, label) in tqdm(enumerate(loader)):
            x = x.to(opt.device)
            # Uses the UPAE to reconstruct the data
            if opt.u:
                out, logvar = model(x)
                rec_err = (out - x) ** 2
                res = torch.exp(-logvar) * rec_err
            # Uses the VAE to reconstruct the data
            else:
                out = model(x)
                rec_err = (out - x) ** 2
                res = rec_err

            if writer and bid == 0:
                writer.add_images(f'test_reconstruction', torch.cat((x, out)).cpu()*0.5+0.5, epoch)
                if opt.u:
                    writer.add_images(f'test_variance', torch.cat((x*0.5+0.5, logvar.exp())).cpu(), epoch)
                writer.add_images(f'test_res', res.cpu(), epoch)

            res = res.mean(dim=(1,2,3))
            # Clamp the abnormality scores to 0-5
            res = torch.clamp(res, 0, 5)
            # Stores the abnormality scores and labels
            y_true.append(label.cpu())
            y_score.append(res.cpu().view(-1))
        
        y_true = np.concatenate(y_true)
        y_score = np.concatenate(y_score)
        # Calculates the AUC
        auc = metrics.roc_auc_score(y_true, y_score)
        print('AUC', auc)
        if plot:
            pres, sense, spec, f1, error_rate = metrics_at_eer(y_score, y_true)
            # # Save the metrics at models/model_results.json by appending
            with open(f"models/model_results.csv", "a") as f:
                f.write(f"{opt.exp},{auc},{pres},{sense},{spec},{f1},{error_rate}\n")

            # Plot the histogram
            plt.hist(y_score[y_true == 0], bins=30,
                     density=False, color='blue', alpha=0.5)
            plt.hist(y_score[y_true == 1], bins=30,
                     density=False, color='red', alpha=0.5)
            # Create legend
            labels = ['Normal', 'Abnormal']
            plt.legend(labels)
            plt.title("Histogram of Abnormality Scores")
            plt.xlabel("Abnormality Score")
            plt.savefig(f"images/{plot_name}_hist.png", dpi=300, bbox_inches='tight')
            plt.clf()

            fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
            plt.plot(fpr, tpr)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.savefig(f"images/{plot_name}_fprtpr.png", dpi=300, bbox_inches='tight')
            plt.clf()
        return auc

# Calculates the metrics at the equal error rate
def metrics_at_eer(y_score, y_true):
    # Gets the false positive rate, true positive rate, and thresholds
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    idx = None
    min_diff = float('inf')
    # Finds the threshold that minimizes the difference between the false positive rate and true positive rate
    for i in range(len(fpr)):
        fnr = 1 - tpr[i]
        diff = abs(fpr[i] - fnr)
        if diff <= 5e-3:
            idx = i
            break
        elif diff < min_diff:
            min_diff = diff
            idx = i
    assert idx is not None
    # Calculates the metrics
    t = thresholds[idx]
    y_pred = np.zeros_like(y_true)
    y_pred[y_score < t] = 0
    y_pred[y_score >= t] = 1
    # Calculates the metrics
    pres = metrics.precision_score(y_true, y_pred)
    sens = metrics.recall_score(y_true, y_pred, pos_label=1)
    spec = metrics.recall_score(y_true, y_pred, pos_label=0)
    f1 = metrics.f1_score(y_true, y_pred)
    print('Error rate:{}'.format(fpr[idx]))
    print('Precision:{} Sensitivity:{} Specificity:{} f1:{}\n'.format(
        pres, sens, spec, f1))
    return pres, sens, spec, f1, fpr[idx]
