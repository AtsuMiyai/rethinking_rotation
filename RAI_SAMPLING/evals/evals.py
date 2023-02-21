import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from scipy.stats import entropy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss().to(device)

softmax = nn.Softmax(dim=0).to(device)


def test_classifier(P, model, test_loader):
    # Switch to evaluate mode
    model.eval()
    correct = 0
    test_size = P.infer_size
    for images, labels in tqdm(test_loader, desc='Test', position=0):
        images = images.to(device)
        labels = labels.to(device)
        images_rot = torch.cat([torch.rot90(images, rot, [2, 3]) for rot in range(4)])
        rot_labels = torch.cat([torch.ones_like(labels) * rot for rot in range(4)], 0)
        output = model(images_rot)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(rot_labels.view_as(pred)).sum().item()
    acc = correct/(test_size*4) * 100
    return acc


def calculate_score(P, model, test_loader):
    
    model.eval()
    score_list = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Test', position=0):
            batch_size = images.shape[0]
            images = images.to(device)
            labels = labels.to(device)
            batch_ent_list = []
            images_rot = torch.cat([torch.rot90(images, rot, [2, 3]) for rot in range(4)]) 
            output = model(images_rot)
            for rot_label in range(4):
                instance_ent_list = []
                for num in range(batch_size): # calculate the entropy per sample.
                    instance_prob = softmax(output[rot_label*batch_size + num])
                    instance_prob = instance_prob.detach().cpu()
                    instance_entropy = entropy(instance_prob)
                    instance_ent_list.append(instance_entropy.item())
                batch_ent_list.append(instance_ent_list)  # 4*batch_size
            batch_score_list = np.mean(batch_ent_list, axis=0)  # batch_size          
            score_list.extend(batch_score_list)
    return score_list