# Annotator bias in Natural Language Understanding datasets

This repository contains the accompanying code for the paper:

**"Are We Modeling the Task or the Annotator? An Investigation of Annotator Bias in Natural Language Understanding Datasets."** Mor Geva, Yoav Goldberg and Jonathan Berant. *In EMNLP-IJCNLP, 2019*.
[https://arxiv.org/abs/1908.07898](https://arxiv.org/abs/1908.07898)

In this work, we investigate whether prevalent crowdsourcing practices for building NLU datasets introduce an "annotator bias" in the data that leads to an over-estimation of model performance.

In this repository, we release our code for:
 * Fine-tuning BERT on the three datasets considered in the paper (experiments 1-3)
 * Converting the three datasets considered in the paper into the annotator recognition task format (experiment 2, this is done as part of the fine-tuning scripts)
 * Generating annotator-based splits (experiment 3)


Please note that the data splits are generated randomly, therefore, reproducing the exact results in the paper is not possible as there might be some variation (see the standard deviation values reported in the paper).

Our experiments were conducted in a **python 3.6.8** environment with **tensorflow 1.11.0** and **pandas 0.24.2**.

## Generation of annotator-based data splits
We considered three NLU datasets in our experiments:
* [MultiNLI](https://www.nyu.edu/projects/bowman/multinli/) (MNLI)
* [OpenBookQA](http://data.allenai.org/OpenBookQA) (OBQA)
* [CommonsenseQA](https://www.tau-nlp.org/commonsenseqa) (CSQA)

The script `obqa_create_data_splits.py` contains the code used for generating annotator-based splits of OBQA and their corresponding random splits. The spliting method is exactly the same for MNLI and CSQA. However, we do not provide the scripts for MNLI and CSQA since annotator information is not publicly available for these datasets (see "note on data availability" below).

**Note on data availability**: At the time of publishing this work, annotator IDs were publicly available only for OBQA. For MNLI and CSQA, this information was not available as part of the official releases. If you are interested in the annotation information, please contact the creators of these datasets.

#### Example commands
1. Generating three annotator-based data splits for each annotator of the top 5 annotators of OBQA:
   ```bash
   python obqa_create_data_splits.py \
   --only_annotator \
   --repeat=3
   ```
   In order to generate both annotator-based data splits and corresponding random splits of the same size, run this command with the value of `only_annotator` set to `False`.

2. Generating three 20%-augmented annotator-based data splits for each annotator of the top 5 annotators of OBQA:
   ```bash
   python obqa_create_data_splits.py \
   --augment_ratio 0.2 \
   --only_annotator \
   --repeat=3
   ```

3. Generating three series of random splits corresponding to the 0%-10%-20%-30%- augmented annotator-based splits of the top 5 annotators of OBQA:
   ```bash
   python obqa_create_data_splits.py \
   --augment_random_series \
   --only_random \
   --repeat=3
   ```


## Model fine-tuning
In all our experiments, we used the pretrained BERT-base cased model from [Google's official repository](https://github.com/google-research/bert).
The directory `model_fine_tuning_scripts` contains the scripts used for fine-tuning and execution of the model on data splits of three datasets considered in the paper.
The table below indicates the experiments covered by each fine-tuning script:

|Fine-tuning script | dataset | utility of annotator information | annotator recognition | generalization across annotators |
|--------|:--------:|:--------:|:-----:|:-----:|
| `run_mnli.py` | MNLI | V | V | V |
| `run_openbookqa.py` | OBQA | V |  | V |
| `run_openbookqa_recognition.py` | OBQA |  | V |  |
| `run_commonsense_qa.py` | CSQA | V |  | V |
| `run_commonsense_qa_recognition.py` | CSQA |  | V |  |


The scripts follow the exact same format as the fine-tuning scripts provided in [Google's official repository](https://github.com/google-research/bert#fine-tuning-with-bert), and should be executed from its root path.

Before running the scripts, make sure you generated the relevant data split files. The directory containing these files should be passed to the argument `data_dir` when running any of these scripts (see examples below).


#### Example commands
1. Fine-tuning the model on the original split of OBQA with annotator IDs concatenated to each example, to test the utility of annotator information:
   ```bash
   export BERT_BASE_DIR=/path/to/bert/cased_L-12_H-768_A-12
   export DATA_DIR=/path/to/obqa/data/splits/dir

   python run_openbookqa.py \
   --do_train=true  --do_eval=true  \
   --data_dir=$DATA_DIR  --vocab_file=$BERT_BASE_DIR/vocab.txt \
   --bert_config_file=$BERT_BASE_DIR/bert_config.json \
   --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
   --max_seq_length=128  --train_batch_size=10  \
   --learning_rate=2e-5  --num_train_epochs=3.0 \
   --output_dir=$BERT_BASE_DIR/openbookqa_with_annotator/ \
   --split=with_annotator
   ```
   To fine-tune on the same data split without annotator IDs, replace the value of the `split` argument with `without_annotator`.

2. Fine-tuning the model to predict annotator IDs of the top 5 annotators of OBQA (annotator recognition):
   ```bash
   export BERT_BASE_DIR=/path/to/bert/cased_L-12_H-768_A-12
   export DATA_DIR=/path/to/obqa/data/splits/dir

   python run_openbookqa_recognition.py \
   --do_train=true  --do_eval=true  \
   --data_dir=$DATA_DIR  --vocab_file=$BERT_BASE_DIR/vocab.txt \
   --bert_config_file=$BERT_BASE_DIR/bert_config.json \
   --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
   --max_seq_length=128  --train_batch_size=10  \
   --learning_rate=2e-5  --num_train_epochs=3.0 \
   --output_dir=$BERT_BASE_DIR/openbookqa_annotator_recognition/ \
   --split=without_annotator
   ```

3. Fine-tuning the model on the top annotator split of OBQA, to test model generalization from all other annotators:
   ```bash
   export BERT_BASE_DIR=/path/to/bert/cased_L-12_H-768_A-12
   export DATA_DIR=/path/to/obqa/data/splits/dir

   python run_openbookqa.py \
   --do_train=true  --do_eval=true  \
   --data_dir=$DATA_DIR  --vocab_file=$BERT_BASE_DIR/vocab.txt \
   --bert_config_file=$BERT_BASE_DIR/bert_config.json \
   --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
   --max_seq_length=128  --train_batch_size=10  \
   --learning_rate=2e-5  --num_train_epochs=3.0 \
   --output_dir=$BERT_BASE_DIR/openbookqa_annotator_0/ \
   --split=annotator  --annotator_idx=0
   ```
   To fine-tune on the corresponding random split of the top annotator, simply replace the value of the `split` argument with `rand`.
   Similarly, to fine-tune on the second multi-annotator split, replace the values of the `split` and `annotator_idx` arguments with `annotator_multi` and `1`, respectively.

4. Fine-tuning the model on the 20%-augmented data split of the third top annotator of OBQA:
   ```bash
   export BERT_BASE_DIR=/path/to/bert/cased_L-12_H-768_A-12
   export DATA_DIR=/path/to/obqa/data/splits/dir

   python run_openbookqa.py \
   --do_train=true  --do_eval=true  \
   --data_dir=$DATA_DIR  --vocab_file=$BERT_BASE_DIR/vocab.txt \
   --bert_config_file=$BERT_BASE_DIR/bert_config.json \
   --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
   --max_seq_length=128  --train_batch_size=10  \
   --learning_rate=2e-5  --num_train_epochs=3.0 \
   --output_dir=$BERT_BASE_DIR/openbookqa_annotator_0/ \
   --split=annotator  --annotator_idx=2  \
   --augment_ratio=0.2
   ```

## Re-producing our experiments on a new dataset

To reproduce our experiments on a crowdsourced NLU dataset for which annotation information is available (i.e. for every example there is an identifier of the annotator who created it), one needs the following for each of our three experiments.
1. **The utility of annotator information**
   * The original data split of the dataset.
   * The original data split of the dataset with the annotator ID concatenated as an additional feature to every example.
   * Fine-tuning script for BERT, suitable for original the dataset task.
2. **Annotator recognition**
   * The original data split of the dataset, with annotator IDs of the top 5 annotators as lables. Namely, every example `(x,y)` written by annotator `z` should be replaced with `(x,z*)`, where `z*=z` if `z` is in the top 5 annotators and `z*=OTHER` otherwise.
   * Fine-tuning script for BERT, for classification task.
3. **Model generalization across annotators**
   * Annotator-based data splits and corresponding random splits of the same size.
   * Augmented annotator-based data splits and corresponding random splits of the same size.
   * Fine-tuning script for BERT, suitable for the original dataset task.


## Citation
If you make use of our work in your research, we would appreciate citing the following:


> @InProceedings{GevaEtAl2019,
  title = {{Are We Modeling the Task or the Annotator? An Investigation of Annotator Bias in Natural Language Understanding Datasets}},
  author = {Geva, Mor and Goldberg, Yoav and Berant, Jonathan},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and 9th International Joint Conference on Natural Language Processing},
  note = {arXiv preprint arXiv:1908.07898},
  year = {2019}
}
