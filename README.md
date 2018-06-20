# Code-Mixed-Dialog
This repository contains the dataset and baseline implementations for the paper ["A Dataset for Building Code-Mixed Goal Oriented Conversation Systems."](https://arxiv.org/abs/1806.05997)

There is an increasing demand for goal-oriented conversation systems which can assist users in various day-to-day activities such as booking tickets, restaurant reservations, shopping, etc. Most of the existing datasets for building such conversation systems focus on monolingual conversations and there is hardly any work on multilingual and/or code-mixed conversations. Such datasets and systems thus do not cater to the multilingual regions of the world, such as India, where it is very common for people to speak more than one language and seamlessly switch between them resulting in code-mixed conversations. For example, a Hindi speaking user looking to book a restaurant would typically ask, *"Kya tum is restaurant mein ek table book karne mein meri help karoge?"*("Can you help me in booking a table at this restaurant?").To facilitate the development of such code-mixed conversation models, we build a goal-oriented dialog dataset containing code-mixed conversations. Specifically, we take the text from the DSTC2 restaurant reservation dataset and create code-mixed versions of it in Hindi-English, Bengali-English, Gujarati-English and Tamil-English. We also establish initial baselines on this dataset using existing state of the art models like sequence-to-sequence and Hierarchical Recurrent Encoder-Decoder models. The dataset and baseline implementations are provided here.

## The Dataset
The dialogue data for English and code-mixed Hindi, Bengali, Gujarati and Tamil are provided in the `data` directory. The respective native language directories also contain the splits of the vocabulary into:
* English Words
* Native Language Words
* Other Words (Named Entities)

Such a split serves as annotation of the words as being code-mixed or belonging to the native language.

## The Baselines
There are two baseline models for this dataset:
* Sequence-to-Sequence with Attention ([Bahdanau et al., 2015](https://arxiv.org/pdf/1409.0473.pdf))
* Hierarchical Recurrent Encoder-Decoder ([Serban et al., 2015](https://arxiv.org/pdf/1507.04808.pdf))

### Dependencies
* [tqdm](https://github.com/tqdm/tqdm)
* [Tensorflow](https://www.tensorflow.org/) version 1.2
* [Pandas](https://pandas.pydata.org/)

### Preprocessing
The baseline models  are provided in the code directory. Before running them you need to preprocess the data using the `preprocess.py` file in the respective baseline directory. The preprocessing is different for both the baselines. You need to provide the source directory in which the train, dev and test data files are and the target directory where the preprocessed files will be dumped:

* `python preprocess.py --source_dir ../../data/hindi --target_dir ../../data/hindi`

### Training

The models can be trained using `train_seq2seq.py` and `train_hred.py` files in the code directory. The arguments required are:
* config_id: The experiment number.
* data_dir: The directory in which the preprocessed files are dumped (The `target_dir` in preprocessing step)
* infer_data: The dataset split(train, dev or test) on which inference should be performed.
* logs_dir: The directory in which log files should be dumped.
* checkpoint_dir: The directory in which model checkpoints should be stored.
* rnn_unit: The cell type (GRU or LSTM) to be used for the RNNs.
* learning_rate: The initial learning rate for Adam.
* batch_size: The mini batch size to be used for optimimzation.
* epochs: The maximum number of epochs to train.
* max_gradient_norm: The maximum norm of the gradients to be used for gradient clipping.
* dropout: The keep probability of RNN units.
* num_layers: The number of layers of RNN to be used for encoding.
* word_emb_dim: The size of the word embeddings to be used for input to the RNN.
* hidden_units: The size of RNN cell hidden units.
* eval_interval: The number of epochs after which validation is to be performed on the dev set.
* patience: The patience parameter for early stopping.
* train: To run the model in train mode or test mode. `True` means train mode is on.
* debug: To run the code in debug mode or not. In debug mode the code runs on a smaller dataset (67 examples) for only 2 epochs. `True` means debug mode is on.

To run the training:
* `python train_seq2seq.py --config_id 1 --data_dir ../data/hindi --infer_data test --logs_dir logs --checkpoint_dir checkpoints --rnn_unit gru --learning_rate 0.0004 --batch_size 32 --epochs 50 --max_gradient_norm 5 --dropout 0.75 --num_layers 1 --word_emb_dim 300 --hiden_units 350 --eval_interval 1 --patience 5 --train True --debug False`

### Testing
To just run inference on the test set use the `train` flag as `False` : 
* `python train_seq2seq.py --config_id 1 --data_dir ../data/hindi --infer_data test --logs_dir logs --checkpoint_dir checkpoints --rnn_unit gru --learning_rate 0.0004 --batch_size 32 --epochs 50 --max_gradient_norm 5 --dropout 0.75 --num_layers 1 --word_emb_dim 300 --hiden_units 350 --eval_interval 1 --patience 5 --train False --debug False`

### Evaluation
The file `get_scores.py` in the `scores` directory produces the BLEU (moses and pycoco), ROUGE, per-response accuracy and the per-dialogue accuracy. We used the BLEU scripts from [Google's seq2seq repo](https://github.com/google/seq2seq/tree/master/seq2seq/metrics) for the moses BLEU and the scripts from [Microsoft COCO Caption Evaluation](https://github.com/tylin/coco-caption/tree/master/pycocoevalcap) for pycoco BLEU. It requires the following 3 arguments:
* --preds_path: The directory where the inference on the test set has dumped its predictions file and labels file.
* --config_id: The experiment number which is appended to the predictions' filename and labels' filename.
* --lang: Can be one of 'english', 'hindi', 'bengali', 'gujarati' and 'tamil'.


