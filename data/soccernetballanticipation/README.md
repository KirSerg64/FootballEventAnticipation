# Setting up SoccerNet Ball Action Anticipation

This directory contains the splits for the Soccernet Ball Action Anticipation dataset. To download the dataset and exract frames use the setup_dataset_BAA.py script. Instructions for using the script are in the main README

To train FAANTRA for the challenge split, modify `train.json` and `val.json` for `train_challenge.json` and `val_challenge`. 

After downloading and extracting frames, the frame naming convention is as follows:

```
data-folder
└───train
    └───clip_1
    |frame0.jpg
    |frame1.jpg
    |...
    └───clip_101
    |frame0.jpg
    |frame1.jpg
    |...
└───valid
    └───clip_1
    |frame0.jpg
    |frame1.jpg
    |...
    └───clip_101
    |frame0.jpg
    |frame1.jpg
    |...
```

---
