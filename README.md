# Developing a Speech Recognition Model Using Convolutional Neural Networks

🚦⚠️👷‍♂️🏗️ Repo Under Construction 🚦⚠️👷‍♂️🏗️

> This repository is based on the liveProject [Recognize Speech Commands with Deep Learning](https://www.manning.com/liveproject/recognize-speech-commands-with-deep-learning) by [Manning Publications](https://liveproject.manning.com/), for which I served as the Implementer.

*If you prefer reading from Jupyter Notebooks, please click on [dl-for-speech-recognition.ipynb](https://github.com/koushikvikram/speech-recognition-deep-learning/blob/main/dl-for-speech-recognition.ipynb).*

---------------------------------------------------------------------

## Contents

- [Introduction]()
- [References]()
- [Importing Necessary Packages]()

## Introduction

| ![](images/ivr.webp) |
|:--:|
| *Credit: https://getvoip.com/blog/2021/03/08/what-is-ivr/*  |
| |


When you dial your Bank's customer service, instead of a human picking up your call, a computer answers and asks you to speak out your account number. You tell your account number. Then, the computer gives you a few options, asks you for more details and transfers your call to a human according to your response. This process is called Interactive Voice Response (IVR). But, have you ever wondered how the computer is actually able to understand what you said?

Enter ASR. IVR uses a technology called Automatic Speech Recognition or ASR, also known as Speech-to-Text (STT) to understand your speech. ASR takes in the audio, converts it to text and passes it on for the computer to make a decision. 

> *"ASR is widely used in contact centres across the globe to enhance customer satisfaction, cutting costs, improving productivity and so on. It is now extremely easy for callers to input different types of information including reason of calling, account numbers, names via the capabilities of a voice recognition technology. All these can be done without any interaction with a live call centre agent. It impacts tremendously on the productivity of contact centre as callers need not have to remain idle while their calls are put on hold since agents are busy with some other calls. This is how a contact centre can do great cost cutting by making effective use of speech recognition technology."* - Speech Recognition And Customer Experience, Sumita Banerjee [[1](https://www.c-zentrix.com/blog/speech-recognition-and-customer-experience)]

Some speech recognition systems require "training" (also called "enrollment") where an individual speaker reads text or isolated vocabulary into the system. The system analyzes the person's specific voice and uses it to fine-tune the recognition of that person's speech, resulting in increased accuracy. Systems that do not use training are called "speaker-independent" systems. Systems that use training are called "speaker dependent". [[2](https://en.wikipedia.org/wiki/Speech_recognition)]

In this project, we'll build a "speaker-independent" system and train it on Google Brain's [Speech Commands Data Set v0.01](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data). [[3](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data)]. Then, we'll test our model on a phone number spoken in our voice. We'll do all of these in a [Kaggle notebook](https://github.com/koushikvikram/speech-recognition-deep-learning/blob/main/dl-for-speech-recognition.ipynb) to take advantage of easy data access and the computing power of GPUs.

## Speech Recognition Process

The entire process is described in the infographic below.

|                                                                |                                                                |
|:--------------------------------------------------------------:|:--------------------------------------------------------------:|
|![Speech Recognition Deep Learning Process](images/process.png) |![Speech Recognition Deep Learning Process](images/audio-dl-viz.png)|


```python

```

## References
1. [Speech Recognition And Customer Experience - C-Zentrix](https://www.c-zentrix.com/blog/speech-recognition-and-customer-experience)
2. [Speech Recognition - Wikipedia](https://en.wikipedia.org/wiki/Speech_recognition)
3. [Speech Commands Data Set v0.01](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data)