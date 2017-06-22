# Modeling Trajectory with Recurrent Neural Networks

The source code of my paper.

Hao Wu, Ziyang Chen, Weiwei Sun, Baihua Zheng, Wei Wang, Modeling Trajectories with Recurrent Neural Networks. IJCAI 2017.

## Environment
- Python: py2/py3 compatible
- Tensorflow version: 12.0 (You may have to change some code if you want to use higher version of Tensorflow since some APIs have been changed after v1.0)
- OS: Linux (I've tried this code on Windows and there may occur some strange runtime problems.)


## Directory Structure
The directory structure may be as follows
```
workspace (e.g., /data)
    └ dataset_name (e.g., porto_6k)
        ├ data
        |   └ your_traj_file.txt
        ├ map
        |   ├ nodeOSM.txt
        |   └ edgeOSM.txt
        └ ckpt
            └ CSSRNN
                ├ dest_emb
                |   └ emb_50_hid_50_deep_1
                ├ dest_coord
                |   └ emb_200_hid_50_deep_1
                └ without_dest
                    └ emb_250_hid_350_deep_3
   
  
codespace
    ├ config
    ├ main.py
    ├ geo.py
    ├ trajmodel.py
    └ ngram_model.py
    
```

## Usage
Tobe completed.
