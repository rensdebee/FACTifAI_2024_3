## Downloading and Pre-processing Datasets

For each dataset, use the provided preprocessing script for each of the train, validation, and test scripts as follows:

```bash
python preprocess.py --split train
python preprocess.py --split val
python preprocess.py --split test
```

For Waterbirds-100%, the dataset needs to be first downloaded using the following [link](https://drive.google.com/file/d/1zJpQYGEt1SuwitlNfE06TFyLaWX-st1k/view?usp=sharing).

## Acknowledgements

The scripts provided here build upon scripts from [stevenstalder/NN-Explainer](https://github.com/stevenstalder/NN-Explainer) and [spetryk/GALS](https://github.com/spetryk/GALS)