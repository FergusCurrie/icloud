# icloud syncer sorter extractor etc.

Needs a .env file containing the following:

- 'APPLE_ID'
- 'APPLE_PASSWORD'

# ml pipeline

model pipeline:

1. pretrainned feature extraction
2. (optional) pretrain with ssl on my data
3. label some data with voxel51
4. supervised finetune on labelled data
5. evaluation on test set

day by day:

1. poll icloud for new images
2. download new images
3. extract features with fineted model
4. classify with knn
5. move to appropriate folder

ideas:

- MLflow for tracking experiments + model registry
- voxel51 for labelling. active learning?
- email feedback
- drift detection
- two models? one for vertical and one for horizontal? or just some hearty centercropping?

# todo

- [x] standardise image types - for now jpg only
- [ ] standardise image sizes
- [x] decide on classes i want to sort into
- [ ] deal with data duplicates. stop data leakage

# labels

- screen shots
- photo of me
- photo with people
- photo of animal photo of scenery
- photo of book
- photo containing text
- default an other catagory?
- food
- my fitness pal
- photo of beer can
- photos inside
-

# masks

- segmentation masks of me

# setup

1. downloading icloud photos to local disk. only downloads undownloaded photos.

```
python3 src/main.py
```

2. generating embeddings for current model

3. classifying images, for current model, with knn? or just

```

```

# diagram
