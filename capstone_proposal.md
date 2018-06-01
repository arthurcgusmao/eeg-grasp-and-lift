# Machine Learning Engineer Nanodegree

## Capstone Proposal
Arthur Colombini Gusmão  
December 29th, 2017; updated on June 1st, 2018.

## Proposal
<!-- _(approx. 2-3 pages)_ -->

This proposal has been retrieved from [a Kaggle competition on EEG recordings](https://www.kaggle.com/c/grasp-and-lift-eeg-detection).

### Domain Background
<!-- _(approx. 1-2 paragraphs)_ -->

<!-- In this section, provide brief details on the background information of the domain from which the project is proposed. Historical information relevant to the project should be included. It should be clear how or why a problem in the domain can or should be solved. Related academic research should be appropriately cited in this section, including why that research is relevant. Additionally, a discussion of your personal motivation for investigating a particular problem in the domain is encouraged but not required. -->

Brain-machine interfaces (BMIs) are equipments that use signals recorded from the brain to drive external hardware. One type of such signals are EEG (electroencephalography). An interesting property of EEG is that it is typically noninvasive: electrodes are placed along the scalp, in constrast with ECoG (electrocorticography), where electrodes are placed directly on the exposed surface of the brain. In clinical contexts, EEG refers to recordings of the brain's spontaneous electrical activity over a period of time, where multiple electrodes are placed on different positions on the scalp.

Currently, the relationship between brain activity and EEG signals is poorly understood. It is believed that a better understanding of these signals, possibly with the help of smart machine learning algorithms, can be used to build smarter BMIs. Better equipments, in turn, can aid patients who have gone through amputation or neurological disabilities to move through the world with greater autonomy.


### Problem Statement
<!-- _(approx. 1 paragraph)_ -->

<!-- In this section, clearly describe the problem that is to be solved. The problem described should be well defined and should have at least one relevant potential solution. Additionally, describe the problem thoroughly such that it is clear that the problem is quantifiable (the problem can be expressed in mathematical or logical terms) , measurable (the problem can be measured by some metric and clearly observed), and replicable (the problem can be reproduced and occurs more than once). -->

The problem this project addresses is how to build a model or algorithm that is able to capture the relationship between brain activity and EEG signals in such a way that building effective BMIs for disabled patients can become feasible. Useful BMIs should have the property that it can predict with high accuracy what the patient is thinking (in our context, what movement he is trying to make). The range of allowed movements should be fixed so that a classifier can map the input (EEG signals up to the current time) to a set of allowed movements, possibly assigning a probability for each class of movement.




### Datasets and Inputs
<!-- _(approx. 2-3 paragraphs)_

In this section, the dataset(s) and/or input(s) being considered for the project should be thoroughly described, such as how they relate to the problem and why they should be used. Information such as how the dataset or input is (was) obtained, and the characteristics of the dataset or input, should be included with relevant references and citations as necessary It should be clear how the dataset(s) or input(s) will be used in the project and whether their use is appropriate given the context of the problem. -->

The data contains EEG recordings of subjects performing graps-and-lift (GAL) trials. The following video shows an example of a trial:

[![Example of recorded movement](https://img.youtube.com/vi/y3_Izuop2gY/0.jpg)](https://youtu.be/y3_Izuop2gY)

There are 12 subjects in total, 10 series of trials for each subject, and approximately 30 trials within each series. The number of trials varies for each series.

Each GAL should correspond to one of 6 events:

1. HandStart
2. FirstDigitTouch
3. BothStartLoadPhase
4. LiftOff
5. Replace
6. BothReleased

These events always occur in the same order. In the training set, there are two files for each subject + series combination:

- the *_data.csv files contain the raw 32 channels EEG data (sampling rate 500Hz)
- the *_events.csv files contains the ground truth frame-wise labels for all events

The events files for the test set are not provided and must be predicted. Each timeframe is given a unique id column according to the subject, series, and frame to which it belongs. The six label columns are either zero or one, depending on whether the corresponding event has occurred within ±150ms (±75frames). A perfect submission will predict a probability of one for this entire window.

#### Important Note

When predicting, data from the future may NOT be used! In other words, when predicting labels for id subj1_series9_11, any frame after 11 of series 9 from subject 1 may not be incorporated. In the real application for of a detection algorithm like this, future data doesn't exist.

Data leakage must not be included. For example, one may not center the signals for subj1_series9_11 based on the mean of all frames for subj1_series9. Instead, one must use the mean based on frame 0 to 10.

Data from other subjects and series outside of the series for which you are predicting may be used for training. For example, one can use all of subj2_series6 when predicting subj1_series5.

The columns in the data files are labeled according to their associated electrode channels. The spatial relationship between the electrode locations (see diagram below) may be considered for the analysis.

![](./images/EEG_Electrode_Numbering.jpg)


A detailed account of the data can be found in
```
Luciw MD, Jarocka E, Edin BB (2014) Multi-channel EEG recordings during 3,936 grasp and lift trials with varying weight and friction. Scientific Data 1:140047. www.nature.com/articles/sdata201447
```






### Solution Statement
<!-- _(approx. 1 paragraph)_

In this section, clearly describe a solution to the problem. The solution should be applicable to the project domain and appropriate for the dataset(s) or input(s) given. Additionally, describe the solution thoroughly such that it is clear that the solution is quantifiable (the solution can be expressed in mathematical or logical terms) , measurable (the solution can be measured by some metric and clearly observed), and replicable (the solution can be reproduced and occurs more than once). -->

Any supervised machine learning algorithm can be used to tackle this problem. The kind of model to be adopted, however, may drastically impact in the final results (the test accuracy). Due to the high dimensionality of the data, the nonlinearities that EEG signals may present, the spatial relationship and the sequential nature of the problem, we intend to use deep neural networks as the model of choice.

Specifically, architectures such as the ones used in recurrent neural networks and convolutional neural networks may be particularly useful. Also, the approximately 3600 trials should be a sufficient number of examples to train a neural network with some complexity.


### Benchmark Model
<!-- _(approximately 1-2 paragraphs)_

In this section, provide the details for a benchmark model or result that relates to the domain, problem statement, and intended solution. Ideally, the benchmark model or result contextualizes existing methods or known information in the domain and problem given, which could then be objectively compared to the solution. Describe how the benchmark model or result is measurable (can be measured by some metric and clearly observed) with thorough detail. -->

Lots of models can be found at [the Kaggle competition webpage](https://www.kaggle.com/c/grasp-and-lift-eeg-detection/leaderboard). From there, we can see that the best results are around 0.98 for the mean columnwise area under receiver operating characteristic curve.

Since in this project we intend to use deep neural networks, it will be interesting to compare the result both with similar and different models. From the competition's webpage, we see models that use neural networks, such as [an CNN model](https://www.kaggle.com/anlthms/convnet-0-89) and [a simpler one](https://www.kaggle.com/bitsofbits/naive-nnet), for instance, and models that use other methods, such as [this model based on SVMs](https://www.kaggle.com/karma86/rf-lda-lr-v2-1) and [this model based on a mixture of classifiers](https://www.kaggle.com/mostafafr/rf-lda-lr-v2-1).


### Evaluation Metrics
<!-- _(approx. 1-2 paragraphs)_

In this section, propose at least one evaluation metric that can be used to quantify the performance of both the benchmark model and the solution model. The evaluation metric(s) you propose should be appropriate given the context of the data, the problem statement, and the intended solution. Describe how the evaluation metric(s) are derived and provide an example of their mathematical representations (if applicable). Complex evaluation metrics should be clearly defined and quantifiable (can be expressed in mathematical or logical terms). -->

The metric this project will focus on is the mean column-wise AUC, that is, the mean of the individual areas under the ROC curve for each predicted column. The reason for this choice is that the mean column-wise AUC discourages the model from not being very discriminative. This is specially valuable when one class (one type of movement) occurs for a very long period of time. If we were to use accuracy instead of AUC, the model could likely start to predict that in most instances (or most periods of time) one class of movement is the most likely to occur, not really discriminating between the possible movements the person is making. Further, all models in the Kaggle competition were already evaluated under this metric, making it easy to compare the model's performance.


### Project Design
<!-- _(approx. 1 page)_ -->

<!-- In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project. -->

As already mentioned, the first model we will experiment with are neural networks. Before that, however, we must explore and understand the data. An important thing to do is to look at the label balance. We must make sure that there is not a significant imbalance in order to be confident that cost functions such as the [sigmoid cross-entropy](https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits) will be effective.

Next, we are going to test different architecture types. We will start with feedforward neural networks before moving into more elaborate architectures. After assessing the initial model's performance, we are going to experiment with convolutional layers. They will be designed to take advantage of the temporal characteristic of the input variables. Of course, we could easily employ *recursive neural networks* (RNNs) model the same phenomenon, but recently convolutional networks (ConvNets) have been shown to achieve state-of-art results and be comparable to RNNs ([Borovykh et al., 2017](https://arxiv.org/pdf/1703.04691.pdf)). Furthermore, ConvNets [can be faster](https://medium.com/@TalPerry/convolutional-methods-for-text-d5260fd5675f) and therefore will allow us to have a feedback on model performance in less time, accelerating the development cycle.

After the architecure is fixed, we are going to tune the hyper-parameters. At this stage, we are going to adopt the following strategy:

- Start with some acceptable values for all hyper-parameters.
- Tune the mini-batch size first. The values considered for the mini-batch size usually vary between 32 and 256.
- Use an adaptative learning rate, through the [tensorflow's AdamOptimizer](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer), with the default initial hyper-parameters. We may experiment with different learning rates, however, the default parameters usually work well in this implementation.
- For the number of epochs, early stopping may be used. Early stopping is when we stop the learning process when the validation error starts increasing, given some tolerance measure.
- The starting number of hidden layers in the model will depend on the architecture adopted. It has been observed that up to 3 hidden layers tend to work very good in fully connected networks, while more than that does not help much. When it comes to ConvNets tough, the more hidden layers there are the better the algorithm tends to perform in general, at least in image classification problems. When dealing with convolutions for time series classification this may not hold, and it is something we are going to experiment with.
- The gap between the validation and training error will guide us on the number of hidden units in each layer (and for the number of hidden layers as well). The adoption of techniques such as dropout or L2 regularization will also influence on that. A good rule for the number of hidden units is to increase them until the validation error stops increasing.
- More guidelines on how to tune the hyperparameters can also be found [here](http://neuralnetworksanddeeplearning.com/chap3.html#how_to_choose_a_neural_network's_hyper-parameters).

If we find that the number of examples could be a factor that is limiting the algorithm from capturing the nature of the problem, we can go for a smaller model and maybe apply dimensionality reduction on the data. Algorithms such as Principal Component Analysis (PCA) can then be used. At first, however, we will not want to perform this sort of strategy because we do not know if it will be necessary and if it can somehow diminish the capability of the model to learn a useful representation. Further, it has been shown that ad-hoc feature extraction can be redundant for physiology-based modeling ([Martinez et al., 2013](https://ieeexplore.ieee.org/document/6496209/)). We hope that the convolutional layers will be capable of dealing well with data without the use of these strategies.

If, after all different architectures tried and hyperparameteres extensively tuned, we are still unable of achieving a good performance with neural networks (at least similar to the baselines presented above), other algorithms will be tried. However, as we saw in the competition's webpage, there are good results with neural networks, so this should not be the case.

### References

[weblink] Grasp-and-Lift EEG Detection, Kaggle Competition. Accessed on 01/06/2018. URL: [https://www.kaggle.com/c/grasp-and-lift-eeg-detection](https://www.kaggle.com/c/grasp-and-lift-eeg-detection)

Anastasia Borovykh, Sander Bohte and Cornelis W. Oosterlee. "Conditional Time Series Forecasting with Convolutional Neural Networks." arXiv:1703.04691, 2017. URL: [https://arxiv.org/pdf/1703.04691.pdf](https://arxiv.org/pdf/1703.04691.pdf).

Hector P. Martinez, Yoshua Bengio and Georgios N. Yannakakis. "Learning deep physiological models of affect," in IEEE Computational Intelligence Magazine, vol. 8, no. 2, pp. 20-33, May 2013. doi: 10.1109/MCI.2013.2247823. URL: [http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6496209&isnumber=6496193](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6496209&isnumber=6496193)

Michael A. Nielsen, "Neural Networks and Deep Learning", Determination Press, 2015.
URL: [http://neuralnetworksanddeeplearning.com/chap3.html#how_to_choose_a_neural_network's_hyper-parameters](http://neuralnetworksanddeeplearning.com/chap3.html#how_to_choose_a_neural_network's_hyper-parameters)

Tal Perry. "Convolutional Methods for Text", Medium (website). Accessed on 01/06/2018. URL: [https://medium.com/\@TalPerry/convolutional-methods-for-text-d5260fd5675f](https://medium.com/@TalPerry/convolutional-methods-for-text-d5260fd5675f)


<!--

**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced? -->
