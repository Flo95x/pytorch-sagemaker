# Deploys model on sagemaker
sh create_model_tar_gz.sh
# start deployment from root dir
cd deployment
python3 upload_model_tar_gz.py
python3 deploy.py
cd ..