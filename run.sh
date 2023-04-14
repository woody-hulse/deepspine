#!/bin/sh
python train.py --config config/synthetic_20200216/alpha/many-to-many/gru_100percentTraining.json
python train.py --config config/synthetic_20200216/alpha/many-to-many/deepSpine_100percentTraining.json

python train.py --config config/synthetic_20200306/alpha/many-to-many/gru_100percentTraining.json
python train.py --config config/synthetic_20200306/alpha/many-to-many/deepSpine_100percentTraining.json
python train.py --config config/synthetic_20200306/beta/many-to-many/gru_100percentTraining.json
python train.py --config config/synthetic_20200306/beta/many-to-many/deepSpine_100percentTraining.json

python test.py --config config/synthetic_20200306/alpha/many-to-many/gru_100percentTraining.json --iteration 999
