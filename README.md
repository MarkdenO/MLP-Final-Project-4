# MLP-Final-Project-4


# Installation
In order to run the project, first make sure you are on python version 3.9 or newer, then install the required packages using:
```
bash requirements.sh
```

Then create the csv including the features for training:
```
python3 preprocess.py --data lid_spaeng --output processed_data
```

Then train the model:
```
python3 classify.py --mode train --data processed_data --output models --class-weights
```

And generate a labeled test set using:
```
python3 classify.py --mode predict --data processed_data --output models --test-file lid_spaeng/test.conll --prediction-output predictions.conll
```
