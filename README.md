# TimeSeries Sandbox


This repo contains materials for the "Modeling timeseries and sequence data on AWS using Apache
MXNet and Gluon" workshop at [Applied ML Days](https://www.appliedmldays.org/) 2018. In this
workshop we learn how to model generic sequence data and timeseries data to do sequence prediction,
forecasting future timeseries, anomaly detection to name a few. We will use Apache MXNet a scalable
deep learning framework to train our neural networks. Once trained we will explore various scalable
deployment options on AWS for both real-time and batch inference.

## Slide
refer to slides for an introduction in to [time series modeling on
AWS](https://www.slideshare.net/mallyajiggs/time-series-modeling-workd-amld-2018-lausanne) 

## Agenda

0. [Introduction to ApacheMXNet and Gluon - Crash Course](intro_mxnet_gluon)
1. [Introduction to TimeSeries Analysis](intro_to_timeseries.ipynb)
2. [Univariate TimeSeries](univariate_timeseries_forecasting_lstm.ipynb)
3. [Multivariate TimeSeries Forecasting](multivariate_timeseries_forecasting.ipynb)
4. [TimeSeries Forecasting with DeepAR](sagemaker-timeseries)
5. [Seq2Seq with Apache MXNet](seq2seq)

## Prerequisites

- Setup an [AWS account](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-account.html)
- Option 1: Run on [Amazon
  SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-setup-working-env.html)
- Option 2: Setup a notebook server on [AWS DL
  AMI](https://s3.amazonaws.com/smallya-test/strata-london/NotebookSetupAWS.html)

## Apache MXNet cheat sheet and Gluon  

- [MXNet Cheat Sheet](https://bit.ly/2xTIwuj)
- [Gluon Tutorial](https://github.com/zackchase/mxnet-the-straight-dope)

# WSAE-LSTM




Repository that aims to implement the WSAE-LSTM model and replicate the results of said model as defined in *"A deep learning framework for financial time series using stacked autoencoders and long-short term memory"* by Wei Bao, Jun Yue, Yulei Rao (2017).

https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0180944

This implementation of the WSAE-LSTM model aims to address potential issues in the implementation model as defined by Bao et al. (2017) while also simultaneously addressing issues in previous attempts to implement and replicate results of said model (i.e. [mlpanda/DeepLearning_Financial](https://github.com/mlpanda/DeepLearning_Financial)). 


## Source journal (APA)

Bao W, Yue J, Rao Y (2017). "A deep learning framework for financial time series using stacked autoencoders and long-short term memory". PLOS ONE 12(7): e0180944. https://doi.org/10.1371/journal.pone.0180944

<u>Diagram Illustrating the WSAE-LSTM model on an abstract level:</u>

![wsae lstm model funnel diagram](https://github.com/timothyyu/wsae-lstm/blob/master/docs/wsae%20lstm%20model%20funnel%20diagram.png)

### Source journal data (saved into `data/raw` folder as `raw_data.xlsx`):
DOI:10.6084/m9.figshare.5028110
https://figshare.com/articles/Raw_Data/5028110

## Repository structure

This repository uses a directory structured based upon [Cookiecutter Datascience]( http://drivendata.github.io/cookiecutter-data-science/#directory-structure).

Repository package requirements/dependencies are defined in `requirements.txt` for pip and/or `environment.yml` for Anaconda/Conda. 

### `mlpanda/DeepLearning_Financial`:

Repository of an existing attempt to replicate above paper in PyTorch ([mlpanda/DeepLearning_Financial](https://github.com/mlpanda/DeepLearning_Financial)), checked out as a `git-subrepo` for reference in the`subrepos`directory. This repository, `subrepos/DeepLearning_Financial`, will be used as a point of reference and comparison for specific components in `wsae-lstm`.