#!/bin/bash

wget -O saved_model/GQCNN-2.0.zip https://berkeley.box.com/shared/static/j4k4z6077ytucxpo6wk1c5hwj47mmpux.zip

cd saved_model
tar zxvf GQCNN-2.0.zip
mv GQ-Image-Wise GQCNN-2.0
cd ..