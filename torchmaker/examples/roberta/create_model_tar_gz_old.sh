# creates model.tar.gz file
# must include .pth, model.py and config file for execution of inference.py
cd deployment
cp ../../../config.py .
# must be inside dir code. See: https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html?ref=morioh.com&utm_source=morioh.com#using-third-party-libraries
mkdir code
cp requirements.txt code
cp inference.py code
git clone https://huggingface.co/T-Systems-onsite/cross-en-de-roberta-sentence-transformer
cd cross-en-de-roberta-sentence-transformer
#git clone https://huggingface.co/distilbert-base-uncased-distilled-squad
#cd distilbert-base-uncased-distilled-squad
tar zcvf ../model.tar.gz  * #../code/* ../config.py
cd ..
rm -rf cross-en-de-roberta-sentence-transformer
rm config.py
cd ..