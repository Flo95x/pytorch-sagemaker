sudo apt-get update -y
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs git -y

git lfs install

# clone projects also with large files like .bin file containing model weights
git clone https://huggingface.co/T-Systems-onsite/cross-en-de-roberta-sentence-transformer
