# Setting up SoccerNet Ball

This directory contains the splits converted from the original SoccerNet Ball Action Spotting dataset, available at: https://www.soccer-net.org/tasks/ball-action-spotting.

There are two methods to download and extract frames from the dataset. The first is the following:

To download the videos follow instructions provided in [SoccerNet](https://www.soccer-net.org/tasks/ball-action-spotting), and to generate the folder structure for frames, use the provided script [extract_frames_snb.py] adapted from E2E-Spot. Although the script is not in this repository and is instead found in the [T-Deed repository](https://github.com/arturxe2/T-DEED_v2)

To train FAANTRA for the challenge split, modify `train.json` and `val.json` for `train_challenge.json` and `val_challenge`. 

Frames are extracted at a resolution of 796x448, and frame naming convention is as follows:

```
data-folder
└───england_efl
    └───2019-2020
        └───2019-10-01 - Blackburn Rovers - Nottingham Forest
        |frame0.jpg
        |frame1.jpg
        |...
        └───2019-10-01 - Brentford - Bristol City
        |frame0.jpg
        |frame1.jpg
        |...
```

The second method is by using the setup_dataset_BAA.py file. Instructions on how to use it are available in the main README

---
