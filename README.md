# Anomaly detection of submersible pump impellers using CNNs and PyTorch

This is the GitHub repository for Francesco Finucci's project for the exam of Deep Learning and Computer Vision by Prof. Michela Quadrini at the university of Camerino.

The goal of this project was to compute binary classification of submersible pump impellers manufactured by casting iron. The castings needed to be classified either in the "ok" class, meaning that the cast was succesfull, and the "defect" class, that means the opposite.

## How to use
Python 3.8 or above is a requirement.

You can easily install all of the python3 dependencies by running the command
```
pip install -r requirements.txt
```
or by installing the requirements using another python container such as _venv_ or _conda_.

## How to run the code
In order to run the code you can run the .py files in the src folder as such:
```
python src/model.py
python src/tester.py
python src/explainer.py
```

## How to use the model
The model's weights are saved in the model/ folder in the file named
```
model.pth
```
