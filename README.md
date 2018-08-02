# Manuscript-Binarization
A FCN based architecture is first trained to binarize text documents.The model is then saved and used for initialization for training on the dataset of old palm leaf manuscripts. This is done by fixing the first few layers of the model (learning rate is set to zero for the first few layers), then training this dataset on it.(Transfer Learning)

The results of the final model are shown below-
