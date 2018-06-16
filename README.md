# Code-Mixed-Dialog
This repository contains dataset and baseline implementations for the paper "A Dataset for Building Code-Mixed Goal Oriented Conversation Systems."

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

### Preprocessing
The baseline models  are provided in the code directory. Before running them you need to preprocess the data using the `preprocess.py` file in the respective baseline directory. The preprocessing is different for both the baselines. You need to provide the source directory in which the train, dev and test data files are and the target directory where the preprocessed files will be dumped:

* `python preprocess.py --source_dir ../../data/hindi --target_dir ../../data/hindi`
