
#Adapted from github
# https://github.com/zhoubolei/CAM/blob/master/pytorch_CAM.py

#Related to paper
# http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf

from torchvision import transforms, datasets
from torch.autograd import Variable
from torch.nn import functional as F
import torch
import numpy as np
import cv2
from models import ResCNN
import os

# output directory
os.makedirs("CAM_output", exist_ok=True)

# Model used for cam in this case ResCNN model is used
model_path = r"F:\UniversitE\Deep Learning Application Lab\CNN\wandb\run-20250831_205141-urd79uxl\files\ResCNN_CAM.pth"

# dataset
data = datasets.CIFAR10(root="data", train=False, download=True)
class_idx = data.classes
# Select some images
image_indices = [19, 371, 496, 754, 903]

# Separate image and label
images_pil, labels = [], []
for i in image_indices:
    images_pil.append(data[i][0])
    labels.append(data[i][1])

# load pretrained resnet18
net = ResCNN(5) #depth 5 not the best one but work fine and take less time to train 
net.load_state_dict(torch.load(model_path,
                                weights_only=True))
net.eval()

# hook the feature extractor
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

net.res_cnn_block[3].register_forward_hook(hook_feature)

# get the softmax weight
params = list(net.parameters())
weight_softmax = np.squeeze(params[-2].data.numpy())

def returnCAM(feature_conv, weight_softmax, class_idx):
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


def get_predicted_class_id(images_pil):
    
    # preprocessing (Same normalizzation used in training)
    preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    predicted_class = []
    predicted_class_prob = []
    for image in images_pil:
        img_tensor = preprocess(image)
        img_variable = Variable(img_tensor.unsqueeze(0))
        logit = net(img_variable)

        # Get probabilities
        h_x = F.softmax(logit, dim=1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        probs = probs.numpy()
        idx = idx.numpy()
        predicted_class.append(idx[0])
        predicted_class_prob.append(probs[0])
    return predicted_class, predicted_class_prob

predicted_class, predicted_class_prob = get_predicted_class_id(images_pil)


for i in range(len(predicted_class)):
    
    # softmax output for first class predicted
    print('{:.3f} -> {}'.format(predicted_class_prob[i], class_idx[predicted_class[i]]))
    
    # generate class activation mapping for the prediction
    CAMs = returnCAM(features_blobs[i], weight_softmax, [predicted_class[i]])

    # convert PIL to cv2 to save the image
    img_cv = cv2.cvtColor(np.array(images_pil[i]), cv2.COLOR_RGB2BGR)
    height, width, _ = img_cv.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img_cv * 0.5
    cv2.imwrite(('CAM_output/CAM_CIFAR10_' + str(i) + '.jpg'), result)
    print("output CAM.jpg for the top1 prediction: %s" % class_idx[predicted_class[i]])