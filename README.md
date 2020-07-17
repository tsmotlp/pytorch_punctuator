# PyTorch Punctuator
----A basic template for punctuation prediction implemented by PyTorch

## 1. File Description
> **data**
> * **`raw_data`**: the directory to store raw data for training, validation, and testing, containing `train.txt`, `valid.txt`, and `test.txt`, respectively.
> * **`processed_data`**: the directory to store the processed data of `raw_data` by `preprocessing.py`, containing `train.npy`, `valid.npy` and `test.npy`, respectively.
> * **`preprocessing.py`**: python file to split inputs and labels, convert input sentences to index, pad index list to the same length, and save the processed data as `.npy` format.
> * **`dataset.py`**: python script to build pytorch `Dataset` and `DataLoader`.
