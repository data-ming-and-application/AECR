Code of AECR: Alignment Efficient Cross-Modal Retrieval Considering Transferable Representation Learning.

**************************** Requirement ****************************
#requirement python 3.6,pytorch 1.7,cuda 11.2,cudnn 7.6


******************************* USAGE ********************************
data.py  -----The code of dataloader(Parallel, Non-parallel data of Flickr30K and MSCOCO).
evaluation.py  -----The code of evaluation.
loss.py -----The code of loss function.
model.py -----The code of model.
opt.py -----The code of setting hyper parameters.
train.py -----The code of training the model.
vocab.py -----The code of producing a vocabulary.


******************************* Data ********************************
We follow SCAN to obtain image features, which can be downloaded by using:
#wget https://scanproject.blob.core.windows.net/scan-data/data.zip

--the pretrain file should contain:
#"SGRAF_f30k_model_best.pth.tar":pretrained model on Flickr30K
#"SGRAF_coco_model_best.pth.tar":pretrained model on MSCOCO

python train.py

