# Source Code for "Scones: Towards Conversational Authoring of Sketches"
## Paper
Please find our paper [here](https://dl.acm.org/doi/abs/10.1145/3377325.3377485).

## Composition Proposer

### Prerequisites
To train new models and/or run inference on published pre-trained checkpoints for the composition proposer, you will need the following prerequisites:
- Tensorflow 2.3.1
- Tensorflow Estimator 2.3.0
- Huggingface Transformers 3.1.0
- nltk 3.5
- tqdm 4.50.2
- numpy 1.18.5
- Data and Pre-trained GLoVe embedding listed in the section below

### Data
For Training/Eval only: Scones is trained on CoDraw data, which can be downloaded in JSON format from [here](https://drive.google.com/file/d/0B-u9nH58139bTy1XRFdqaVEzUGs/view). Please download the json file and move it into the *data/* folder at the root of the repo

For Training/Eval/Prediction only (generating new scenes based no captions): Scones preprocesses text tokens into GLoVe vector. We use 300-d GLoVe vectors trained on Common Crawl with 42B tokens. Please download the file *glove.42B.300d.zip* from [here](https://nlp.stanford.edu/projects/glove/) and extract the file *glove.42B.300d.txt* into the *data/* folder.

### Pretrained Model
A new pretrained model that uses the huggingface GPT-2 implementation can be downloaded from [here](https://drive.google.com/file/d/1Anny8fyV46jwnXgiJ4YveR0HvcvAdWUP/view?usp=sharing). This model achieves **3.53** for the similarity metric on CoDraw dataset's test set. 

### Training
To train the model, simply run
`python train_state.py`

### Eval
To run evaluation on the test set, change the *model_ckpt* variable in *run_eval.py* to the desired checkpoint path. Then, run:
`python run_eval.py`

### Generation
Coming soon.

## Object Sketchers
Coming soon.