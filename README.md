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

It is worth mentioning that the theory driven nature simplifies the model structure and makes it more appealing and intuitive. Couple of points to stress:
  - The main driver of our model is Sequence model, so it is more convenient to view it from sequence perspective rather than graph. Indeed, as we showed previously,    the main variability is explained by sequence model (~75%) and the rest we hope can be addressed by graph info. This motivates a two-step implementation:
    - run RNN forward pass end-to-end for each individual;
    - scanning through career transition anchor point of each individual to enhance its embedding by metapath guided reference employee and their titles;
This is a very nice structure, because in one model, we achieve two modes, i.e., by remove (b), we recover typically model. Moreover, the interaction of a) and b) is not a + b, but a * b (a compose b). This is very critical, because, from regression we know, a + b > a/b, but a * b is a new factor, if it has better performance than a or b, it really means their combination has new chemistry.
  - The MetaPath guided reference employees/titles can be assembled in advance, and in training time, we only need to run a static graph attention network, which in our case is just a simple aggregation. The trick is to aggregate the ETET paths for a sliding window and apply the limited lookback and non-anticipative constraints, we call it a ETET snapshot. It allows you to find, for each focal employee, the reference employee and title from a particular time instance within the lookback window. Now, due to our theory, we assign different weights to target titles based on the similarity of two employees' dynamic typically. Notice, it is based the dynamic typically NOT dynamic typically + graph embedding, which allows us to easily find the dynamic typically from the RNN pass that has been run already. This is such a great simplification but yet releastic and interpretable. Otherwise, we need to use a temporal metapath guided heterogenous graph network, which really defeats the purpose of this modeling.
 
 For ongoing techinical notes, please see:
 
 https://www.overleaf.com/project/641fb40c74572939b28b2322
 
 For transparencies, please see:
 
 https://www.overleaf.com/project/6415dbed238908944eecffd8
