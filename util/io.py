import os
import json
import numpy as np

FPS_SN = 25 # Remember to change value in dataset.frame if you change it here

def load_json(fpath):
    with open(fpath) as fp:
        return json.load(fp)

# Function to store the prediction of a model in a json file
def store_json_snba(pred_path, pred, pad_idx, class_dict, stride = 1):
    inv_class_dict = {v: k for k, v in class_dict.items()}
    for k,v in pred.items():
        splitDict = dict()
        splitDict['videos'] = []
        # Pad indices that do not predict any actions
        indices_to_pad = v.sum(axis=-1) == 0
        # Find predicted action and its confidence value
        pred = np.argmax(v, axis=-1)
        pred_conf = np.max(v, axis=-1)
        # Replace indices with no actions with padding value
        pred[indices_to_pad] = pad_idx
        # Iterate through all clips
        for clip_num, clip in enumerate(pred):
            clipSum = 0
            clipDict = {"annotations": {"observation": [], "anticipation": []}} # Observation is always empty since we only anticipating
            # Iterate through all predictions and write to dictionary
            for frame, event in enumerate(clip):
                if event != pad_idx:
                    clipSum += 1
                    eventDict = {}
                    position = int(frame / FPS_SN * 1000 * stride)
                    eventDict['label'] = inv_class_dict[event]
                    eventDict['position'] = position+30000
                    eventDict["confidence"] = float(pred_conf[clip_num, frame])
                    eventDict["confidence_vect"] = v[clip_num,frame].tolist()       # Store confidence vector on top of confidence for top class
                    clipDict["annotations"]["anticipation"].append(eventDict)
            if clipSum == 0:
                print(clip_num)
            clipDict["path"] = f"clip_{clip_num+1}"
            splitDict['videos'].append(clipDict)
        # Store prediction dictionary into json file
        path = os.path.join('/'.join(pred_path.split('/')[:-1]) + '/preds', k)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path + '/results_anticipation.json', 'w') as fp:
            json.dump(splitDict, fp, indent=2)

def load_text(fpath):
    lines = []
    with open(fpath, 'r') as fp:
        for l in fp:
            l = l.strip()
            if l:
                lines.append(l)
    return lines