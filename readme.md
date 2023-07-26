# Voice Spoofing Detection Project

## Task

The detection problem is framed as a binary classification task: given a audio, label it as either bonafide or spoof.

## Recommended Datasets

- Training set: [ASVSpoof2019](https://arxiv.org/abs/1911.01601)
- Test set: [ASVSpoof2021](https://arxiv.org/abs/2210.02437)

### Download Dataset

- Training set: [ASVSpoof2019(PA.zip)](https://datashare.ed.ac.uk/handle/10283/3336)
- Test set: [ASVSpoof2021](https://zenodo.org/record/4834716#.Y-2Xh-xBydY)
- Progress
  set: [ASVSpoof2021(progress)](https://cuhko365.sharepoint.com/:u:/t/Researchteam/ERQGQFM--XtIpDTQ_1NVB7oBmdweUbyz8-DIAsIK1vfuSw?e=tfbUH2).

### Download Data Annotation File

[PA keys and metadata](https://www.asvspoof.org/asvspoof2021/PA-keys-full.tar.gz)

### Requirements

[//]: # (Conda)

[//]: # (```)

[//]: # (conda env create -f environment.yaml)

[//]: # (```)
Pip

```
pip install -r requirements.txt
```

### Train, valid and test

The training, validation, and testing code are all in main.py.
A validation is performed after each training epoch, and the model with the highest validation set accuracy is selected
as the test model.
Training and testing are controlled by the is_train parameter.

```
python main.py
```

### Score

The test script generates scores for each test sample and then tests the final results
by [eval-package](https://github.com/asvspoof-challenge/2021/tree/main/eval-package) in ASVSpoof2021 (note: case 1 is
used and the track parameter is specified as PA).

Before running the eval-package, one change is needed, add the following code to **line 639** of
**[main.py](https://github.com/asvspoof-challenge/2021/blob/main/eval-package/main.py)**

```
if subset == 'progress':
    protocol_cm_pd = protocol_cm_pd.loc[score_cm_pd.index]
```

Test on progress subset:

```
python main.py --cm-score-file score_progress.txt --track PA --subset progress
```

Test on eval set:

```
python main.py --cm-score-file score.txt --track PA --subset eval 
```
