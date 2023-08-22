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
    valid_loader = xray_data.get_xray_dataloader(
        opt.batchsize, WORKERS, 'valid', img_size=opt.image_size, dataset=opt.dataset)

    opt.epochs = EPOCHS
    train_loop(model, loader, test_loader, valid_loader, opt)

def train_loop(model, loader, test_loader, valid_loader, opt):
    device = torch.device('cuda:{}'.format(opt.cuda))
    print(opt.exp)

    optim = torch.optim.Adam(model.parameters(), opt.lr, betas=(0.5, 0.999))
    writer = SummaryWriter('log/%s' % opt.exp)

    # Early Stopping Parameters
    epochs_last_improvement = 0
    loss1_best = float('inf')
    best_epoch = 0
    early_stop = False
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
                if opt.loss_type == 'MSE':
                    loss1_pre = mse_loss(x, out)
                    loss1 = torch.mean(loss1)
                elif opt.loss_type == 'SSIM':
                    loss1_pre = ssim_loss(x, out)
                    loss1 = torch.mean(loss1)
                l1s.append(loss1.item())
            # Uses the UPAE to reconstruct the data
            else:
                # mean is the reconstructed data, logvar is the variance
                out, logvar = model(x)
                if opt.loss_type == 'MSE':
                    loss1_pre = mse_loss(x, out)
                    loss1 = torch.mean(torch.exp(-logvar)*loss1_pre)
                elif opt.loss_type == 'SSIM':
                    loss1_pre = ssim_loss(x, out)
                    loss1 = torch.mean(torch.exp(-logvar)*loss1_pre)

                loss2 = torch.mean(logvar)
                loss = loss1 + loss2
                l1s.append(loss1_pre.mean().item())
                l2s.append(loss2.item())

            # Backpropagation
            optim.zero_grad()
            loss.backward()
            optim.step()
        # Tests the data for each epoch
        auc = test_for_xray(opt, model, test_loader, writer=writer, epoch=e)
        # Test data with validation set
        test_for_xray(opt, model, valid_loader, plot_name="valid", writer=writer, epoch=e)

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
            writer.add_scalar('loss1_pre', loss1_pre.mean(), e)
            writer.add_scalar('rec_err', l1s, e)
            writer.add_scalar('logvars', l2s, e)
            writer.add_images('reconstruction', torch.cat((x, out)).cpu()*0.5+0.5, e)
            writer.add_images('variance', torch.cat(
                (x*0.5+0.5, logvar.exp())).cpu(), e)
            print('epochs:{}, recon error:{}, logvars:{}'.format(e, l1s, l2s))

        # Early Stopping
        if loss1_pre.mean() < loss1_best:
            epochs_last_improvement = 0
            loss1_best = loss1
            best_epoch = e
            torch.save(model.state_dict(), 'models/%s.pth' % opt.exp)
        else:
            epochs_last_improvement += 1
            if epochs_last_improvement >= opt.patience:
                early_stop = True
                break

    # Saves the model
    if not early_stop:
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
                if opt.loss_type == 'MSE':
                    loss = mse_loss(x, out)
                    res = torch.exp(-logvar) * loss
                elif opt.loss_type == 'SSIM':
                    loss = ssim_loss(x, out)
                    res = torch.exp(-logvar) * loss
            # Uses the VAE to reconstruct the data
            else:
                out = model(x)
                if opt.loss_type == 'MSE':
                    loss = mse_loss(x, out)
                    res = loss
                elif opt.loss_type == 'SSIM':
                    loss = ssim_loss(x, out)
                    res = loss

            if writer and bid == 0:
                writer.add_images(f'{plot_name}_reconstruction', torch.cat((x, out)).cpu()*0.5+0.5, epoch)
                if opt.u:
                    writer.add_images(f'{plot_name}_variance', torch.cat((x*0.5+0.5, logvar.exp())).cpu(), epoch)
                writer.add_images(f'{plot_name}_res', res.cpu(), epoch)
                writer.add_scalar(f'{plot_name}_rec_err', res.mean(), epoch)

            res = res.mean(dim=(1,2,3))
            # Clamp the abnormality scores to 0-5
            res = torch.clamp(res, 0, 5)
            # Stores the abnormality scores and labels
            y_true.append(label.cpu())
            y_score.append(res.cpu().view(-1))
        
        if plot_name == "valid":
            return
        y_true = np.concatenate(y_true)
        y_score = np.concatenate(y_score)
        auc = metrics.roc_auc_score(y_true, y_score)

        # Add histogram in tensorboard
        if writer:
            writer.add_scalar(f'{plot_name}_auc', auc, epoch)
            # Add y_score histogram where y_true is 0
            writer.add_histogram(f'{plot_name}_y_score_0', y_score[y_true == 0], epoch)
            # Add y_score histogram where y_true is 1
            writer.add_histogram(f'{plot_name}_y_score_1', y_score[y_true == 1], epoch)

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



def s1_score(image_i, image_j, c1=1e-6):
    """Calculates the first term of the SSIM score between image_i and image_j
    Args:
        image_i: The first image
        image_j: The second image
        c1: A constant used to stabilize the division"""
    mu_x_i = image_i
    mu_x_j = image_j
    numerator = 2 * mu_x_i * mu_x_j + c1
    denominator = mu_x_i ** 2 + mu_x_j ** 2 + c1
    s1 = numerator / denominator
    return s1


def compute_covariance_between_images(image_tensor1, image_tensor2, epsilon = 1e-6):
    # Reshape the image tensors to a 2D shape (num_pixels, 1)
    flattened_image1 = image_tensor1.reshape(-1, 1)
    flattened_image2 = image_tensor2.reshape(-1, 1)

    # Center the data
    centered_data1 = flattened_image1 - torch.mean(flattened_image1, dim=0, keepdim=True) + epsilon
    centered_data2 = flattened_image2 - torch.mean(flattened_image2, dim=0, keepdim=True) + epsilon
 
    # Calculate the covariance matrix between the two images
    covariance_matrix = torch.matmul(centered_data1.t(), centered_data2) / (centered_data1.shape[0] - 1)

    return covariance_matrix


def s2_score(image_i, image_j, c2=1e-6):
    """Calculates the second term of the SSIM score between image_i and image_j"""
    covariance = compute_covariance_between_images(image_i, image_j)
    numerator = 2 * covariance + c2
    denominator = torch.var(image_i) + torch.var(image_j) + c2
    s2 = numerator / denominator

    return s2

def ssim_loss(x, mu_x):
    """Calculates the SSIM loss between the original data x and its reconstruction mu_x
    Args:
        x: The original data
        mu_x: The reconstruction of the original data"""
    
    # Calculate the SSIM distance between the original data x and its reconstruction mu_x
    ssim_distance = torch.sqrt(2 - s1_score(x, mu_x) - s2_score(x, mu_x))
    ssim_distance = torch.where(torch.isnan(ssim_distance), torch.zeros_like(ssim_distance) + 0, ssim_distance)
    return ssim_distance


def pixel_wise_logarithm(sigma_x):
    """Calculates the pixel wise logarithm of the variance sigma_x"""
    return torch.log(sigma_x ** 2)

def mse_loss(x, mu_x):
    """Calculates the MSE loss between the original data x and its reconstruction mu_x"""
    return ((x - mu_x) ** 2)