# shapenet

This is a fork from [original repo shapenet](https://github.com/justusschock/shapenet), which contains reference, paper and explanation. 


## Changes from original repo

Changes are mostly about preprocessing training data. Images are saved as numpy binary for faster subsequent execution. And some dependencies are removed so that it is simpler to run. 

## Download training data

Data can be downloaded from [here](http://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz)

## Available command 

To make pca 

```bash
cd shapenet/scripts && python make_pca.py
```

To preprocess data 
```bash
python -m shapenet.scripts.preprocess
```

To train
```bash
python -m shapenet.scripts.train --datadir "/path/to/data"
```

To predict
```bash
python -m shapnet.scripts.predict --datadir "/path/to/data" --img "/path/to/target/image"
```




