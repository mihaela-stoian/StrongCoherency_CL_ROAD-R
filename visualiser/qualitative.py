import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pyplot as pyplot
from matplotlib import patches
from matplotlib import patheffects

classes = ['Ped', 'Car', 'Cyc', 'Mobike', 'MedVeh', 'LarVeh', 'Bus', 'EmVeh', 'TL', \
           'OthTL', 'Red', 'Amber', 'Green', 'MovAway', 'MovTow', 'Mov', \
           'Brake', 'Stop', 'IncatLeft', 'IncatRht', 'HazLit', 'TurLft', 'TurRht', 'Ovtak', \
           'Wait2X', 'XingFmLft', 'XingFmRht', 'Xing', 'PushObj', 'VehLane', 'OutgoLane', 'OutgoCycLane', \
           'IncomLane', 'IncomCycLane', 'Pav', 'LftPav', 'RhtPav', 'Jun', 'XingLoc', 'BusStop', 'Parking']

import torch
import numpy as np


def find_violations(out, Iplus, Iminus, Mplus, Mminus, data_split, model_type, video_id, img_id, threshold=0.5):
    # the commented code is for the not thresholded violations - we are not interested about it atm

    batch_size, num_rules, num_classes = len(out), len(Iplus), len(Iplus[0])

    out = torch.from_numpy(out)
    Iplus = torch.from_numpy(Iplus)
    Iminus = torch.from_numpy(Iminus)
    Mplus = torch.from_numpy(Mplus)
    Mminus = torch.from_numpy(Mminus)

    out = out.unsqueeze(1)
    t_out = out.data > threshold

    if out.shape[0] == 0:
        return torch.zeros([1]), torch.zeros([1])

    t_H = t_out.expand(batch_size, num_rules, num_classes)

    t_impl_value, _ = torch.min(
        (((Iplus * t_H) + (Iminus * (~t_H))) + (1 - Iplus - Iminus)) * (1 - Iplus * Iminus) + torch.min((Iplus * t_H),
                                                                                                        Iminus * (
                                                                                                            ~t_H)) * Iplus * Iminus,
        dim=2)
    t_head_value, _ = torch.max(t_H * torch.transpose(Mplus, 0, 1) + (~t_H) * torch.transpose(Mminus, 0, 1), dim=2)
    t_violations = t_impl_value > t_head_value
    t_num_violations = torch.sum(t_violations)

    # CODE TO WRITE DOWN THE VIOLATIONS !!!!PER PREDICTION!!!!
    # thv stands for thresholded violations

    # with open("violations/"+model_type+"_"+data_split+"_th"+str(threshold)+".txt", 'a') as fwrite:
    #     for t_pred_violations in t_violations:
    #         fwrite.write(str(video_id)+','+str(img_id) +',' + ','.join([str(int(tv.item())) for tv in t_pred_violations])+'\n')

    for t_pred_violations in t_violations:
        if True in t_pred_violations:
            # print(torch.nonzero(t_pred_violations),img_id,video_id)
            return True
    return False




# createMs returns two matrices:
# Mplus: shape [num_constraints, num_labels] --> each row corresponds to a constraint and it has a one if the constraint has positive head at the column number of the label of the head
# Mminus: shape[num_constraints, num_labels] --> each row corresponds to a constraint and it has a one if the constraint has negative head at the column number of the label of the head
def createMs(file_path, num_classes):
    Mplus, Mminus = [], []
    with open(file_path, 'r') as f:
        for line in f:
            split_line = line.split()
            assert split_line[2] == ':-'
            mplus = np.zeros(num_classes)
            mminus = np.zeros(num_classes)
            if 'n' in split_line[1]:
                # one indentified that is negative, ignore the 'n' to get the index
                index = int(split_line[1][1:])
                mminus[index] = 1
            else:
                index = int(split_line[1])
                mplus[index] = 1
            Mplus.append(mplus)
            Mminus.append(mminus)
    Mplus = np.array(Mplus).transpose()
    Mminus = np.array(Mminus).transpose()

    return Mplus, Mminus

def draw_ractangle(img_path, bbox, out, th, found_violations=False):
    # We use batche size = 1
    out = out[0]

    img = matplotlib.image.imread(img_path)
    org_height, org_width = img.shape[:2]

    figure, ax = pyplot.subplots(1)

    # width = float(bbox[2]) - float(bbox[0])
    # height = float(bbox[3]) - float(bbox[1])

    bbox[0] = (float(bbox[0]) / 682) * org_width  # width x1
    bbox[2] = (float(bbox[2]) / 682) * org_width  # width x1
    bbox[1] = (float(bbox[1]) / 512) * org_height  # width x1
    bbox[3] = (float(bbox[3]) / 512) * org_height  # width x1

    rect = patches.Rectangle((float(bbox[0]), float(bbox[1])), float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1]),
                             edgecolor='r' if found_violations else 'g', facecolor="none", linewidth=2)

    # Old incorrect
    # rect = patches.Rectangle((float(bbox[0]),float(bbox[1])),float(bbox[2]),float(bbox[3]), edgecolor='r', facecolor="none", linewidth=2)

    ax.imshow(img)

    print(np.where(out > th)[0])

    labels = [classes[idx] for idx in np.where(out > th)[0]]
    preds = [out[idx] for idx in np.where(out > th)[0]]
    str_labels = '\n'.join([f"{l}" for l, p in zip(labels, preds)])

    # ax.text(float(bbox[0]),(float(bbox[1])-50),str(str_labels),verticalalignment='top',color='white',fontsize=10,weight='bold').set_path_effects([patheffects.Stroke(linewidth=4, foreground='black'), patheffects.Normal()])
    ax.text(float(bbox[0]), (float(bbox[1]) - 10), str(str_labels), multialignment='left', color='white', fontsize=10,
            weight='bold').set_path_effects([patheffects.Stroke(linewidth=4, foreground='black'), patheffects.Normal()])

    ax.add_patch(rect)
    title = img_path.split('/')[-2:]
    plt.title(title)
    plt.tight_layout()
    pyplot.show()
    return figure

