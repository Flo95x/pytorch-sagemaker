# creates model.tar.gz file
# must include .pth, model.py and config file for execution of inference.py
cp ../../config.py .
tar zcvf deployment/model.tar.gz config.py train/cifar_net.pth model.py
rm config.py
