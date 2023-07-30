# icloud sync & sort as mlops practice.

Using TensorFlow Extended, beam, fiftyone, cvat create a ml pipeline for classifying my icloud.

# features

- [x] sync icloud photos to local disk
- [x] manage database with fifty one
- [x] create annotation jobs with cvat
- [ ] create embeddings with qdrant
- [ ] active learning
- [ ] mlflow integraiton
- [x] store data tfrecord
- [ ] finetune model with tfx
- [ ] deploy model with tfx (or cron job)
- [ ] drift detection
- [ ] continual learning

# setup

Needs a .env file containing the following:

- 'APPLE_ID'
- 'APPLE_PASSWORD'

setting up cvat

```
cd /home/fergus/cvat
docker-compose up # can add -d to run in background, but i like to close when not using
```

running qdrant

```
docker run -p 6333:6333 qdrant/qdrant
```

1. downloading icloud photos to local disk. only downloads undownloaded photos.

```
python3 src/main.py
```

2. generating embeddings for current model

3. classifying images, for current model, with knn? or just
