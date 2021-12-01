import torchvision
import torch
import torch.nn.functional as F
import numpy as np


def gradcam(model, img, feature_dimension):
    """Performs Gradcam for a given model and slices
    Args:
        model (model): trained pytorch model
        img (array): CT Slices
        feature_dimension (int, optional): Number of filters for last convolution (Or convolution of interest). Defaults to 512.
    Returns:
        array: GradCam Heatmap
    """

    # Gradient = 0
    model.zero_grad()
    
    # Prediction
    pred, activations = model(img)
    pred = pred.squeeze(0)
    
    # Gradienten berechnen
    pred[0].backward()
    gradients = model.get_activations_gradient()
    pooled_gradients = torch.mean(gradients, dim=[0,2,3])
    
    # Gradcam
    for i in range(feature_dimension):
        activations[:, i, :, :]*= pooled_gradients[i]

    # Heatmap
    heatmap = torch.sum(activations, dim=1).squeeze().detach().cpu()
    heatmap = F.relu(heatmap)
    heatmap -= heatmap.min()
    heatmap /= torch.max(heatmap)
    heatmap = heatmap.detach().cpu().numpy()
    return heatmap


def gradcam_pp(model, img):
    """Performs Gradcam++ for a given model and slices
    Args:
        model (model): trained pytorch model
        img (array): CT Slices
    Returns:
        array: GradCam Heatmap
    """

    # Gradient = 0
    model.zero_grad()
    
    # Prediction
    pred, activations = model(img)
    pred = pred.squeeze(0)
    score = pred[0]
    
    # Gradienten berechnen
    score.backward()
    gradients = model.get_activations_gradient()
    b, k, u, v = gradients.size()

    alpha_num = gradients.pow(2)
    alpha_denom = gradients.pow(2).mul(2) + \
            activations.mul(gradients.pow(3)).view(b, k, u*v).sum(-1, keepdim=True).view(b, k, 1, 1)
    alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))

    alpha = alpha_num.div(alpha_denom+1e-7)
    positive_gradients = F.relu(score.exp()*gradients) # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
    weights = (alpha*positive_gradients).view(b, k, u*v).sum(-1).view(b, k, 1, 1)

    saliency_map = (weights*activations).sum(1, keepdim=True)
    saliency_map = F.relu(saliency_map)
    # saliency_map = F.upsample(saliency_map, size=(224, 224), mode='bilinear', align_corners=False)
    saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
    saliency_map = (saliency_map-saliency_map_min).div(saliency_map_max-saliency_map_min).data

    return saliency_map.cpu().numpy()[0][0]


def scorecam(model, img, device):
    """Computes activation maps by using ScoreCAM
    # https://github.com/haofanwang/Score-CAM/blob/master/cam/scorecam.py


    Args:
        model (torch model): ScoreCAM model
        img (image tensor): Image tensor
        device (device): torch device

    Returns:
        array: ScoreCAM heatmap
    """
    H, W = img.shape[2:]
    model.zero_grad()
    
    img = img.to(device)
    model = model.to(device)
    
    pred, activations = model(img)

    prob = F.softmax(pred, dim=1)
    activations = F.relu(activations)
    activations = F.interpolate(activations, (H, W), mode="nearest")
    B, C, _, _ = activations.shape
    
    score_activation = torch.zeros(1, 1, H, W)

    with torch.no_grad():
        for i in range(C):
            activation_k = activations[:, i, :, :]
            normalized_activation_k = (activation_k - activation_k.min()) / (activation_k.max() - activation_k.min())
            normalized_activation_k.to(device)
            pred2 = model(img * normalized_activation_k)[0]
            prob2 = F.softmax(pred2, dim=1).squeeze(0)

            score = prob2[0].cpu()
            if torch.isnan(score):
                continue
            score_activation += score * activation_k.cpu()
    score_activation = F.relu(score_activation)

    score_activation_norm = (score_activation - score_activation.min()) / (score_activation.max() - score_activation.min())
    #print(score_activation_norm)

    return score_activation_norm.cpu().numpy()[0][0]