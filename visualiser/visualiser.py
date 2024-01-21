import matplotlib.pyplot as plt
import numpy as np
import argparse

from req_handler import createIs, createMs
from qualitative import find_violations, draw_ractangle

AGENTS = ['Ped', 'Car', 'Cyc', 'Mobike', 'MedVeh', 'LarVeh', 'Bus', 'EmVeh', 'TL', 'OthTL']
ACTIONS = ['Red', 'Amber', 'Green', 'MovAway', 'MovTow', 'Mov', 'Brake', 'Stop', 'IncatLft', 'IncatRht', 'HazLit', 'TurLft', 'TurRht', 'Ovtak', 'Wait2X', 'XingFmLft', 'XingFmRht', 'Xing', 'PushObj']
LOC = ['VehLane', 'OutgoLane', 'OutgoCycLane', 'IncomLane', 'IncomCycLane', 'Pav', 'LftPav', 'RhtPav', 'Jun', 'xing', 'BusStop', 'parking']
ALL_LABELS = AGENTS + ACTIONS + LOC

IMG_IDS = ['02339'] #, '05807', '05252', '04317', '00929']
VIDEO_IDS = ['2015-02-03-08-45-10_stereo_centre_04', '2014-06-26-09-31-18_stereo_centre_02']
REQ_LABELS = ['TL']

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('preds_path', type=str)
    parser.add_argument('--th', default=0.5, type=float)
    return parser.parse_args()


def get_positive_labels(preds, threshold=0.5):
    pos_labels = []
    for i, pred in enumerate(preds):
        if pred > threshold:
            pos_labels.append(ALL_LABELS[i])
    return pos_labels


def load_preds(args, preds_path:str, counter_per_img=10):
    # Read constraints from file and create the Ms and Is matrices
    Iplus, Iminus = createIs('../constraints/constraints.txt', 41)
    Mplus, Mminus = createMs('../constraints/constraints.txt', 41)

    counter = 0
    old_id = ''
    with open(preds_path, 'r') as f:
        for line in f:
            counter += 1
            line = line.strip().split(',')
            if line[-1] == '':
                line = line[:-1]
            videoname = line[0]
            img_id = line[1][:-4]
            new_id = videoname+img_id
            if old_id == '':
                old_id = new_id

            if new_id == old_id and counter > counter_per_img:
                continue
            if new_id != old_id:
                counter = 1
                old_id = new_id
            if img_id not in IMG_IDS or videoname not in VIDEO_IDS:
                continue
            bbox_coords = line[2:6]
            preds = np.array(list(map(float, line[7:])))
            pos_labels = get_positive_labels(preds, threshold=args.th)
            if not set(pos_labels).intersection(set(REQ_LABELS)):
                continue

            preds = np.expand_dims(preds, axis=0)

            found_viol = find_violations(preds, Iplus, Iminus, Mplus, Mminus, video_id=videoname, img_id=img_id,
                                         data_split=None, model_type=None, threshold=args.th)
            if pos_labels: # and found_viol:
                fig = draw_ractangle(f"/users-2/mihaela/datasets/road/rgb-images/{videoname}/{img_id}.jpg", bbox_coords, preds, th=args.th, found_violations=found_viol)
            # fig.savefig(f"qualitative_img/{args.model}_{videoname}_{img_id}")


def main():
    args = get_args()
    load_preds(args, args.preds_path)


if __name__ == '__main__':
    main()