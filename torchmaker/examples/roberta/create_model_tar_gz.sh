# creates model.tar.gz file
export model_id=cross-en-de-roberta-sentence-transformer

# place inference py inside of code dir
mkdir $model_id/code
cp deployment/inference.py $model_id/code

# create model.tar.gz
cd $model_id
tar zcvf model.tar.gz *
cd ..
mv $model_id/model.tar.gz .
