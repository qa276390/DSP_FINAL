# DSP_FINAL

## Usage

```shell
pip3 install -r requirements.txt
```

### Folder and Dataset
To start this project we have to download dataset from Google Drive by TA and place to `data` folder.


    .
    ├── ...
    ├── predict.py                # predict val set
    ├── Main.ipynb                # training script for phase spectrum    
    ├── Main-mel.ipynb            # training script for magnitude spectrum            
    ├── Ensemble.ipynb            # ensemble script       
    ├── raw2mel.ipynb             # generate mel scale data       
    ├── utils                   
    │   ├── resnet.py             # torchvision resnet
    │   ├── transform.py          # transform funciton used by data generator
    │   └── plot_cm.py            # plot confusion matrix
    │   └── ...
    ├── data 
    │   ├── train                 # training set
    │   ├── val                   # validation set
    |   └── test.npy              # testing set
    └── ...

### Predict
to generate the best validation accuracy as report. 

```shell
    python3 predict.py
```

### Training
- see `Main.ipynb` for phase spectrum  training.
- see `Main-mel.ipynb` for magnitude spectrum training. Before training, see run `raw2mel.ipynb` to have the training data.
- see `Ensemble.ipynb` to have the final result.
