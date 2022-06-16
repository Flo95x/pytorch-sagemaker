# creates model.tar.gz file
# must include .pth, model.py and config file for execution of inference.py
cp ../../config.py .
cp training/cifar_net.pth .
tar zcvf deployment/model.tar.gz config.py cifar_net.pth model.py
rm cifar_net.pth
rm config.py
