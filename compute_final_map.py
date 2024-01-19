
""" 
 Testing 
"""

import argparse
import os
from tqdm import tqdm
import numpy as np
import pickle as pkl

def main():
    parser = argparse.ArgumentParser(description='Extracting txt preds from pkl file')
    parser.add_argument('filepath', help='path to dir of detections in pkl format, for each frame')
    parser.add_argument('--extra_id', default=True, action="store_false")

    args = parser.parse_args()
    res = 0
    agent, action, loc = 0,0,0
    with open(args.filepath, 'r') as f:
        for line in f:
            #print(line)
            if "Results for test & agent" in line and 'ness' not in line:
             agent=True
             continue 
            if agent:
             if ':' in line:
              res_t = float(line.split(':')[3])
              print(line.split(':')[0], res_t)
              res += res_t
             if 'OthTL' in line: agent = False

            if "Results for test & action" in line:
             action=True
             continue
            if action:
             if ':' in line:
              res_t = float(line.split(':')[3])
              print(line.split(':')[0],res_t)
              res += res_t
             if 'PushObj' in line: action = False

            if "Results for test & loc" in line:
             loc=True
             continue
            if loc:
             if ':' in line:
              res_t = float(line.split(':')[3])
              print(line.split(':')[0],res_t)
              res += res_t
             if 'parking' in line: loc = False; break;

    print('{:.2f}'.format(res/41.0))

if __name__ == "__main__":
    main()
