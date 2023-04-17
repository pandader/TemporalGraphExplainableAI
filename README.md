# Hybrid Dynamic Model To Decipher Job Mobility

In this project, we develop a novel architecture to model dynamic evolution of employee's career trajectory. The objective is to explain the variability in career transition by:

  - Dynamic Typicality
  - Social Influence
 
 The main modules are:
  - Sequence model with Time-Awareness: the intention is to capture the duration and time decay effect when calculating the dynamic typicality of a focal employee;
  - Heterogenous Graph guided by MetaPath: the intention is to leverage on social network to recommend jobs based on the similarity of focal employee and reference employees;
 
 Notice, the main driver is the sequence model that explains the most part of career transitions, the Heterogenous network is only used to explain residuals.
 
 In this repository:
 
  - Notebooks (ipynb): experiment ideas and test apis;
  - dataUtils: data pre-processing, feature engineering and data preparation for learning;
  - Encoders: rnn encoder, graph convolution encoder and metapath driven heterogenous attention network;
  - Decodres: bi-level sequence decoder to decompose the career transition probability into coarse title dist and finer title dist
  - main: the main entry of the training/testing
 
 For ongoing techinical notes, please see:
 
 https://www.overleaf.com/project/641fb40c74572939b28b2322
 
 For transparencies, please see:
 
 https://www.overleaf.com/project/6415dbed238908944eecffd8
