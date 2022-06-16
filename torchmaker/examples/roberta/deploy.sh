
while true; do
    read -p "Do you want to start initial setup? (y/n)" yn
    case $yn in
        [Yy]* ) sh setup.sh; break;;
        [Nn]* ) break;;
        * ) echo "Please answer yes or no.";;
    esac
done

# create model.tar.gz and upload it on aws s3
sh create_model_tar_gz.sh
python3 upload_model_tar_gz.py

# start deployment from root dir
cd deployment
python3 deploy.py
