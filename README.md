# Evaluating Pronominal Anaphora in Machine Translation: An Evaluation Measure and a Test-suite
This repository contains the data and source code of our paper "[Evaluating Pronominal Anaphora in Machine Translation: An Evaluation Measure and a Test-suite](https://arxiv.org/abs/1909.00131)" in EMNLP-IJCNLP 2019.


### Prerequisites

```
* PyTorch 0.4 or higher
* Python 3
* AllenNLP
```

### Dataset - Training/Development/Testing
The training data consists of system translations vs. reference text from WMT 2011-15. There are two versions of the development data: one with unique system translations vs. reference, and one with unique noisy sentences vs. reference. Both are from WMT2014. The test data is the data used for the user studies in French, German, Russian and Chinese. 

Development and test data are uploaded here. The training data and a trained model are available at the following link: https://www.dropbox.com/sh/ol66hb2t3jcdeny/AADrfqm7fH1Cq8uiIvzcUuUra?dl=0

## How To Run
* Training: <br>
```
python train.py [training_data] [dev_data]
```
Other paramaters can be changed in `train.py`.

* Testing: <br>
```
python test_trained_model.py [model] [test_data]
```

### Test Suite
The database with the pronoun test suite contains all the samples for all source languages from WMT2011-2017; filter by source language as required. Among other data, each sample provides the source sentence and two previous sentences for context, the equivalent for reference, and also which system translation error it originated from. This test suite is not filtered by the results of the user study to keep it unrestricted; if required, use the pronoun pairs from the keys in the `pronoun_pair_list.pkl` dictionary file, and filter using the `reference_pronoun` and `system_pronoun` fields. 

## Citation
Please cite our paper if you found the resources in this repository useful.
```
@inproceedings{EvalAnaphora,
  title={Evaluating Pronominal Anaphora in Machine Translation: An Evaluation Measure and a Test Suite},
  author={Prathyusha Jwalapuram and Shafiq Joty and Irina Temnikova and Preslav Nakov},
  booktitle={EMNLP-IJCNLP},
  year={2019}

}	
```
