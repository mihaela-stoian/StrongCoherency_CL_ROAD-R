
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
    parser.add_argument('dirpath', help='path to dir of detections in pkl format, for each frame')
    parser.add_argument('--extra_id', default=True, action="store_false")
    parser.add_argument('--output_dir', type=str, default=None)


    args = parser.parse_args()
    input_path = args.dirpath
    if args.output_dir is not None:
        output_path = args.output_dir
    else:
        output_path = input_path

    if args.extra_id:
        if input_path[-1] == '/':
            extra_id = input_path.split('/')[-3]
        else:
            extra_id = input_path.split('/')[-2]
        output_path = output_path + '/ccn_' + extra_id + '_all_test_detections_output.txt'
    else:
        output_path = output_path + '/all_test_detections_output.txt'
        # output_path = 'all_test_detections_output.txt'
    print(output_path)

    with open(output_path, 'w') as g:
        for path, subdirs, files in os.walk(input_path):
            for name in tqdm(files):
                if name[-4:] == '.txt':
                    continue
                filepath = os.path.join(path, name)
                img_name = filepath.split('/')[-1][:-4] + '.jpg'
                video_name = filepath.split('/')[-2]

                data = pkl.load(open(filepath, 'rb'))
                if 'main' in data:
                    data = data['main']
                    for det in data:
                        det = det[:46]
                        char_array = np.char.mod('%.16f', det)
                        det = ",".join(char_array)
                        g.write(','.join([video_name, img_name, det]) + ',\n')


if __name__ == "__main__":
    main()
