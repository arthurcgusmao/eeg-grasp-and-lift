# EEG Grasp and Lift

This project contains implementation of a convolutional neural network model used to tackle [a Kaggle competition on EEG recordings](https://www.kaggle.com/c/grasp-and-lift-eeg-detection), used as my final project for the Udacity's Machine Learning Nanodegree course.

For a detailed report about the project's definition, data analysis, methodology, results and conclusions, please refer to the [`capstone_report.md`](./capstone_report.md) file.

## License

All code in this repository is distributed under the terms of the GNU General Public License, version 3 (or, at your choosing, any later version of that license).


## Code execution

Code was developed using Python 2.7, tensorflow 1.7.0, scikit-learn 0.19.1, numpy 1.14.0, pandas 0.22.0, matplotlib 2.0.0, and jupyter 1.0.0.

The convolutional neural network can be trained by running the [pipeline.py](./pipeline.py) file. After that, a model will be saved under the `results` directory, in a directory named with the timestamp of when learning started, which we call the model results directory. The model results directory stores information about training (useful for plotting the learning curve), about the model itself, and tensorflow checkpoints and graph data (useful for restoring the model later). Data analysis and results visualization can be seen by running code present in the Jupyter notebooks.
