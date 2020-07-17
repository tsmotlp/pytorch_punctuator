# PyTorch Punctuator
----A basic template for punctuation prediction implemented by PyTorch

## 1. File Description
> **data**
> * **`raw_data`**: the directory to store raw data for training, validation, and testing, containing `train.txt`, `valid.txt`, and `test.txt`, respectively.
> * **`processed_data`**: the directory to store the processed data of `raw_data` by `preprocessing.py`, containing `train.npy`, `valid.npy` and `test.npy`, respectively.
> * **`preprocessing.py`**: python script to split inputs and labels, convert input sentences to indexes, pad index list to the same length, and save the processed data as `.npy` format.
> * **`dataset.py`**: python script to build pytorch `Dataset` and `DataLoader`.

> **models**
> * **`BaseModel.py`**: the father class implementing the network building, setup input, forward computation, backpropagation, network saving and loading, learning rate schedulers, and visualization of losses and metrics.
> * **`**Model.py`**: the implementaion that extends `BaseModel` of specific models (methods), such as `LstmModel`, `Seq2SeqModel` etc.
> * **`**Net.py`**: the code of network achitectures, such as `LstmNet.py`, `Seq2SeqNet.py` etc.

> **run**
> * **`trainer.py`**: a basic template python file for training from scratch, or resuming training, and validation the `**Model`.
> * **`tester.py`**: a basic template python file for testing the `**Model`.

> **utils**
> * **`configs.py`**: the python file can be used to store and modify the hyper-parameters for training, validation and testing process.
> * **`help_functions.py`**: the python file can be used to store and modify the model initilization strategies and optimizer scheduler settings.
> * **`metrics.py`**: the python file can be used to store and modify the evaluation metrics, such as `Precsion`, `Recall`, `F1-score` etc.
> * **`visualizer.py`**: the python file can be used for visualization of the losses and images.
