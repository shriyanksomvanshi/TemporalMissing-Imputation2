
# Brief summary of important papers

Asadi-2019-**A convolution recurrent autoencoder for spatio-temporal missing data imputation**

In this study, the researchers investigated the problem of missing data in spatio-temporal sensor data collected over a large geographical area. They proposed a convolution bidirectional-LSTM model to capture both spatial and temporal patterns and used autoencoders for missing data imputation. Their evaluation using traffic flow data revealed that the convolution recurrent neural network they proposed performed better than existing methods, demonstrating its effectiveness in addressing missing data in spatio-temporal datasets.


Bogaerts--2020-**A graph CNN-LSTM NN for short and long-term traffic forecasting** ❤️

In this study, the researchers developed a deep neural network for traffic forecasting that utilized graph convolution for spatial features and Long Short Term Memory (LSTM) cells for temporal features. They trained and tested the model using sparse trajectory data from DiDi's ride-hailing service in Xi'an and Chengdu, China. Additionally, they introduced a data-reduction technique based on temporal correlation to select relevant road links as input. The combined approach outperformed high-performance algorithms like LSTM and those from the TRANSFOR19 competition, maintaining its accuracy across various time horizons, ranging from 5 minutes to 4 hours, for multi-step traffic predictions.

Chan-2021-**A neural network approach for traffic prediction and routing with missing data imputation for intelligent transportation system** ❤️

In this study, the researchers addressed the challenges of traffic management, focusing on accurate traffic simulation and handling missing data. They proposed three solutions: a realistic traffic simulation model, a pheromone-based neural network traffic prediction and rerouting system, and a weighted missing data imputation method called Weighted Missing Data Imputation (WEMDI). They benchmarked the traffic simulation against Google Maps rerouting, and WEMDI integration improved traffic factors by 38% to 44% compared to no rerouting system, and up to 19.39% over the base rerouting system for 50% missing data. The WEMDI system also demonstrated robustness in routing other locations, exhibiting high performance.

Chan-2023-**Missing Traffic Data Imputation for Artificial Intelligence in ITS** ❤️

In this research, the investigators addressed the issue of missing data in Intelligent Transportation Systems (ITS) and its potential impact on traffic data analysis. They conducted a comprehensive review of popular AI-based methods for missing data imputation in the context of traffic. The study standardized missing data terminology, discussed limitations of using existing datasets for urban traffic research, and explored statistical and data-driven imputation methods. Their findings revealed that tensor decomposition-based methods were the most commonly used, followed by Generative Adversarial Networks and Graph Neural Networks, all relying on large training datasets. Probability Principle Component Analysis (PPCA) methods were also noted for real-time traffic imputation. Additionally, the study emphasized the importance of more efficient and reliable traffic data collection methods, such as online APIs.

Cui-2020-**Graph Markov network for traffic forecasting with missing data** ❤️

In this study, the researchers addressed the challenge of missing traffic data in the context of short-term traffic forecasting. They treated the traffic network as a graph and formulated the transition between traffic states as a graph Markov process, allowing them to infer missing data while considering spatial-temporal relationships among road links. They proposed a new neural network architecture called the graph Markov network (GMN) and a spectral graph version (SGMN) using spectral graph convolution. Comparative experiments with real-world traffic data demonstrated that GMN and SGMN outperformed baseline models in terms of prediction accuracy and efficiency. The study also provided comprehensive analysis and visualization of model parameters, weights, and predicted results.

Cui-2020-**Stacked bidirectional and unidirectional LSTM recurrent neural network for forecasting network-wide traffic state with missing values** ❤️

In this study, the researchers focused on improving short-term traffic forecasting using recurrent neural networks (RNNs). They introduced a stacked bidirectional and unidirectional LSTM network architecture (SBU-LSTM) to enhance the predictive power of spatial-temporal data and handle missing values in traffic datasets. The bidirectional LSTM (BDLSTM) was utilized to capture temporal dependencies in both directions, and a data imputation mechanism (LSTM-I) was designed to infer missing values and aid in traffic prediction. Experiments with real-world traffic data demonstrated that the proposed SBU-LSTM, particularly the two-layer BDLSTM network, outperformed other models in terms of accuracy and robustness when dealing with various patterns of missing data.

Deng-2022-**Graph Spectral Regularized Tensor Completion for Traffic Data Imputation** 🚩

In this study, the researchers addressed the challenge of incomplete traffic data in intelligent transportation systems (ITS) caused by sensor malfunctions and communication faults. They introduced a novel approach that modeled the road network's topology as a graph and used graph Fourier transform (GFT) to process traffic data. They applied graph-tensor singular value decompositions (GT-SVD) to extract spatial information and developed a graph spectral regularized tensor completion algorithm with temporal constraints. Extensive experiments on real traffic datasets showed that their approach outperformed existing methods, achieving higher accuracy in recovering missing traffic data under various missing patterns.

Jiang-2021-**Imputation of Missing Traffic Flow Data Using Denoising Autoencoders** ❤️

In this study, the researchers used Denoising Autoencoders to address missing traffic flow data in transportation engineering. They trained these models on data with a high missing rate of about 80% and found that even under extreme conditions, the Autoencoder models remained robust, maintaining accuracy. They compared three types of Autoencoders (Vanilla, CNN, and Bi-LSTM) and found that Vanilla performed well even with a high missing rate, while CNN was less suitable for data imputation. Bi-LSTM showed potential for improvement but was computationally expensive. The study also revealed error patterns for different sensor stations and times, with higher errors on weekends. By separating data into weekdays and weekends for training and testing, they significantly improved imputation accuracy. This separation method has the potential for accuracy enhancement in traffic flow data imputation.

Jiang-2022-**A Deep Learning Framework for Traffic Data Imputation Considering Spatiotemporal Dependencies** 🚩

In this study, the researchers tackled the issue of missing or incomplete spatiotemporal data, particularly in the context of traffic analysis. They proposed a novel spatiotemporal data imputation model that effectively captured both spatial and temporal dependencies. Their model utilized temporal convolution and self-attention networks to capture long-term and dynamic spatial dependencies. Additionally, the model incorporated self-learning node embeddings to understand the intrinsic attributes of different sensors. Comparing their model to benchmark algorithms (BTMF and LRTC-TNN) on real-world datasets, they found that their approach outperformed existing methods in most cases, demonstrating its efficacy in spatiotemporal data imputation tasks. However, they noted that model performance could be affected by the amount of available historical data.

Khan-2019-**Development and Evaluation of Recurrent Neural Network-Based Models for Hourly Traffic Volume and Annual Average Daily Traffic Prediction** ❤️

In this study, the researchers aimed to predict high-resolution hourly traffic volumes using robust recurrent neural network (RNN)-based forecasting models. They addressed the challenges of missing data in the dataset by employing two approaches: masking and imputation, in combination with RNN models. Three RNN units, including simple RNN, gated recurrent unit (GRU), and long short-term memory (LSTM), were utilized to develop and evaluate forecasting models. The analysis revealed that the LSTM model outperformed the others, and imputation proved more effective than masking for predicting future traffic volume. The LSTM-Median model was identified as the best overall model for accurately predicting annual average daily traffic (AADT) and hourly traffic volume, capturing long-term seasonal variations in the time-series data while maintaining high accuracy. The study highlights the potential for further research to expand upon these results and explore other RNN models and imputation methods for improved accuracy.

Li-2019-**Missing Value Imputation for Traffic-Related Time Series Data Based on a Multi-View Learning Method** 🚩

In this study, the researchers addressed the challenge of missing sensor readings on highways and their impact on traffic monitoring and data mining. They proposed a multi-view learning method that combined data-driven algorithms (long short-term memory and support vector regression) with collaborative filtering techniques to estimate missing values in traffic-related time series data. The model considered both local and global variations in temporal and spatial views to capture more information from the existing data. Evaluation on highway network data showed that their approach outperformed other baseline methods, especially for block missing patterns with high missing ratios. The study demonstrated the effectiveness of multi-view learning in traffic data imputation and highlighted the model's robustness across various missing scenarios. However, the authors suggested future improvements, such as incorporating non-recurrent traffic conditions and considering the influence of intersections in urban areas.

Li-2022-**Dynamic adaptive generative adversarial networks with multi-view temporal factorizations for hybrid recovery of missing traffic data** ❤️

![image](https://github.com/shriyanksomvanshi/TemporalMissingImputation/assets/143463033/14aa3b4b-5fa5-452a-a07e-440afdd31698) 
`Figure 1: Spatial–temporal (ST) hybrid architecture of the TFs-DGAN model`

In this study, the researchers aimed to improve the imputation of missing traffic data by addressing the challenges of spatial-temporal correlations and heterogeneity. They proposed a hybrid framework called TFs-DGAN, consisting of a dynamic adaptive generative adversarial network (DA-GAN) and multi-view temporal factorizations (TFs). DA-GAN was designed to generate traffic data from noise distribution while TFs refined imperfections by modeling multi-view temporal properties. They evaluated TFs-DGAN on real traffic datasets with missing rates ranging from 10% to 99.99% and various missing patterns (RM, CM, HM). The results demonstrated that TFs-DGAN consistently outperformed state-of-the-art baseline models in terms of accuracy, stability, and computational efficiency. Additionally, TFs-DGAN exhibited good performance even in extreme cases of high missing rates. Visual verification further confirmed the model's effectiveness in imputing missing traffic data. The researchers concluded that TFs-DGAN represents a promising approach for handling missing traffic data in intelligent transportation systems. Future work may focus on incorporating additional factors like weather and further optimizing the model's structure and efficiency.

Liang-2021-**Dynamic spatiotemporal graph CNN for traffic data imputation with complex missing patterns** ❤️

The researchers addressed the problem of missing traffic data in intelligent transportation systems by proposing a novel deep learning framework called Dynamic Spatiotemporal Graph Convolutional Neural Networks (DSTGCN). DSTGCN combines recurrent and graph-based convolutional layers to capture spatiotemporal dependencies in traffic data. Additionally, a graph structure estimation technique is introduced to model dynamic spatial dependencies. Extensive experiments using real traffic datasets with various missing patterns demonstrated that DSTGCN outperformed existing deep learning models in imputing missing traffic data, especially when complete training data was available. The study also compared DSTGCN with a tensor factorization model and found that their performance varied depending on the availability of training data and the complexity of the missing patterns. The research highlights the potential of deep learning models for traffic data imputation and suggests future directions for applying these techniques in intelligent transportation systems.

Liang-2022-**Memory-augmented dynamic graph convolution networks for traffic data imputation with diverse missing patterns** ❤️

![image](https://github.com/shriyanksomvanshi/TemporalMissingImputation/assets/143463033/d0f3a94a-5776-4a71-b3ca-48f1fe4c36d9)

`Figure 2: Types of missing data`


In their research, the scientists addressed the challenge of missing data in traffic data collection for intelligent transportation systems. They introduced a novel deep learning framework called Memory-augmented Dynamic Graph Convolution Networks (MDGCN) to impute missing traffic data. This framework utilized a recurrent layer to capture temporal information and a graph convolution layer to capture spatial information. To overcome existing limitations, they introduced an external memory network to share global spatiotemporal information and a technique to learn dynamic spatial dependencies from traffic data. Their experiments on public traffic speed datasets demonstrated that MDGCN outperformed other deep learning approaches across various missing data scenarios, highlighting the effectiveness of their proposed methods.

Liang-2023-**Spatial-Temporal Aware Inductive Graph Neural nw for C-ITS Data Recovery** ❤️

![image](https://github.com/shriyanksomvanshi/TemporalMissingImputation/assets/143463033/1167a711-5cc0-4a91-9517-415926f7f360)

`Figure 3: The demonstration of data recovery workflow in cooperative
intelligent transportation system`

![image](https://github.com/shriyanksomvanshi/TemporalMissingImputation/assets/143463033/33dd8e38-c06f-45d4-b12b-08dd74cc43f2)
`Figure 4: Data missing patterns of traffic data`

### (a) Random Missing is caused by unexpected transmission errors, and interpolation methods can quickly fill the missing values. 

### (b) Segment Missing is caused by power outages, sensor malfunctioning, and extreme weather conditions. Factorization-based methods and neural network-based models can fill these missing values. 
### (c) Blockout Missing is caused by new deployments or long-time failure, as filling missing values for such situations may be challenging, given that no historical data is available, so thus, nearby sensors are used to fill the need to handle the complicated spatial-temporal dependencies.

In their study, the researchers tackled the challenge of imputing missing entries in spatial-temporal traffic data for Intelligent Transportation Systems (ITS). They introduced a unified model called Spatial-Temporal Aware Data Recovery Network (STAR), which leveraged Graph Neural Networks (GNNs) for real-time and inductive inference. The STAR model utilized a residual gated temporal convolution network to learn temporal patterns and an adaptive memory-based attention model to capture spatial correlations. Through extensive experiments on real-world datasets, they found that STAR consistently outperformed other methods by 1.5-2.5 times in various imputation tasks, supported data recovery for extended time periods, and demonstrated robust performance in transfer learning and time-series forecasting, making it a valuable tool for Cooperative-ITS data recovery needs.

![image](https://github.com/shriyanksomvanshi/TemporalMissingImputation/assets/143463033/2e5a4002-5dd9-4eaf-87f4-ee3ccd89898e)
`Figure 5: Framework of STAR`


Ming-2022-**Multi-Graph Convolutional Recurrent Network for Fine-Grained Lane-Level Traffic Flow Imputation** 

![image](https://github.com/shriyanksomvanshi/TemporalMissingImputation/assets/143463033/5fc5f709-fc83-4673-ab1c-99cf563f207a)
`Figure 6: An illustrative example of road network`

In this study, researchers developed a Multi-graph Convolutional Recurrent network (MACRO) framework to enhance lane-level traffic flow imputation for Intelligent Transportation Systems. They designed a spatial dependency module to capture diverse spatial correlations within traffic flows using multi-relation graphs and multi-graph convolution neural networks. Additionally, a modified bi-directional recurrent neural network addressed temporally continuous data gaps, and a spatio-temporal knowledge integration module improved traffic flow imputation. The experiments on real-world data demonstrated that MACRO outperformed several state-of-the-art methods in traffic flow imputation.

![image](https://github.com/shriyanksomvanshi/TemporalMissingImputation/assets/143463033/9a970fb5-c3e7-4d6f-8dec-07c96879124b)
`Figure 7: The diagrammatic sketch of the proposed MACRO framework for
lane-level traffic flow imputation` 


Najafi-2020- **Estimation of Missing Data in Intelligent Transportation System** 🚩

In this research, the scientists addressed missing data challenges in traffic speed and travel time estimations within Intelligent Transportation Systems (ITS). They focused on a machine learning-based approach called Multi-Directional Recurrent Neural Network (M-RNN) to handle missing data due to sensor instability and communication errors. The study utilized a TomTom dataset from the Greater Toronto Area and found that M-RNN significantly outperformed existing methods, reducing Root Mean Square Error (RMSE) by up to 58%.

![image](https://github.com/shriyanksomvanshi/TemporalMissingImputation/assets/143463033/5fe8c2fa-e574-4f58-b7b4-ede21ff62859)
`Figure 8: System Pipeline for ITS in the Setup`

Pamula-2019-**Impact of Data Loss for Prediction of Traffic Flow on an Urban Road Using Neural Networks** ❤️

In this study, the researchers explored traffic data prediction in Intelligent Transportation Systems (ITS) with missing or inaccurate data. They evaluated the performance of two neural network types, namely a multilayer perceptron (MLP) and a deep learning network (DLN) based on autoencoders, for predicting traffic parameters. The research found that the DLN, despite its complexity, outperformed MLP in handling missing or erroneous data, demonstrating its potential for efficiently assessing traffic conditions and aiding in ITS decision-making. The study used a dataset from the Greater Toronto Area, focusing on traffic flow measurements from ten sites, and considered spatio-temporal relationships in the data.

![image](https://github.com/shriyanksomvanshi/TemporalMissingImputation/assets/143463033/7ae5d0e3-cb65-4dde-ac30-e92ea0c9600e)
`Figure 9: Multilayer perceptron for traffic flow prediction`

![image](https://github.com/shriyanksomvanshi/TemporalMissingImputation/assets/143463033/e7ac8797-0464-40f2-ab65-ead4fe3084c0)
`Figure 10: Deep learning network based on autoencoders for prediction`

Ran-2016-**Tensor based missing traffic data completion with spatial–temporal correlation** ❤️

In this study, researchers addressed the challenge of missing and suspicious traffic data in intelligent transportation systems. They introduced a novel tensor-based method that considered both spatial and temporal information from multiple detecting locations to impute missing traffic flow data. Their approach, utilizing a low-n-rank tensor completion algorithm, successfully improved imputation performance, particularly in extreme cases of extended data absence. The study emphasized the importance of incorporating spatial information for more accurate imputation, and while the experiments were conducted on similar traffic flow data from a freeway corridor, the authors acknowledged the need for further research on diverse traffic flow scenarios and variables in urban settings.

![image](https://github.com/shriyanksomvanshi/TemporalMissingImputation/assets/143463033/6eca3eee-0fa7-434c-bc1d-b2f7cfe786c0)
`Figure 9: The tensor representation of traffic flow data: Tensor representation can capture the hidden weekly, daily, spatial correlation of traffic flow data`

Shen-2023-**Bidirectional spatial–temporal traffic data imputation via graph attention recurrent neural network** ❤️

In this study, researchers addressed the challenge of incomplete spatiotemporal traffic data in intelligent transportation systems (ITS). They introduced a novel approach called the graph attention recurrent neural network (GARNN) for traffic data imputation. The method incorporated both temporal and spatial perspectives by using separate LSTMs for observations and missing data, employing a decay mechanism and graph attention network (GAT) to capture interdependencies across time steps and spatial correlations. The evaluation on two public datasets and three different missing scenarios demonstrated that the GARNN model outperformed other baseline methods, highlighting its effectiveness in handling missing traffic data in ITS applications.

Sun-2022-**Traffic Missing Data Imputation- A Selective Overview of Temporal Theories and Algorithms** ❤️

In this review, the researchers examined the challenge of missing traffic data in intelligent transportation systems (ITS). They analyzed various methods for imputing missing temporal traffic data, considering factors like research methods, missing patterns, assumptions, and application conditions. After testing five representative methods on California PeMS data, they found that probabilistic principal component analysis (PPCA) performed the best under most conditions. The review highlighted the progress in temporal imputation strategies and called for further exploration of spatial and spatial-temporal imputation models in future research.

Tak-2016-**Data-Driven Imputation Method for Traffic Data in Sectional Units of Road Links** ❤️

In this study, the researchers addressed the challenge of missing data imputation in intelligent transportation systems, specifically for sections of road. They proposed a data-driven imputation method that considered spatial and temporal correlations between multiple sensors within a section, improving computational efficiency and preserving the geometrical properties of each section. Comparative analysis demonstrated that their modified k-nearest neighbor approach outperformed other methods like nearest historical data and expectation maximization across various missing data types, missing ratios, day types, and traffic states, showing accurate and stable imputation performance, particularly when dealing with mixed or unidentified missing data types.

Tang-2020-**Missing data imputation for traffic flow based on combination of fuzzy neural network and rough set theory** ❤️

In this study, the researchers addressed the challenge of missing traffic flow data in intelligent transportation systems (ITS) by proposing a hybrid method that combined fuzzy rough set (FRS) and fuzzy neural network (FNN). They used FNN for data classification, followed by the K-Nearest Neighbor (KNN) method to determine optimal data for estimating missing values in each category. Finally, fuzzy rough set was used for imputing missing data. Evaluation using RMSE, R, and RA indicators demonstrated that the proposed hybrid method outperformed traditional approaches such as average-based and regression-based methods for traffic flow data with varying time intervals and missing ratios, confirming its effectiveness and validity in traffic data imputation for ITS applications.

Wang-2022-**A Hybrid Data-Driven Framework for Spatiotemporal Traffic Flow Data Imputation** ❤️

In this study, the researchers aimed to improve the estimation of missing traffic flow data, which is crucial for urban planning and intelligent transportation systems. They introduced a hybrid missing data imputation framework called ST-PTD, which integrated periodic patterns using time-series analysis and described traffic flow trends using novel matrix decomposition. They also utilized a dendritic neural network to fuse the periodic and trend characteristics of missing data. The results, based on actual traffic flow data from Wuhan, China, demonstrated that the ST-PTD framework outperformed eight existing methods in terms of imputation accuracy, highlighting its effectiveness in enhancing data quality for complex traffic flow patterns.

Wang-2022-**A multi-view bidirectional spatiotemporal graph network for urban traffic flow imputation** ❤️

In this study, the researchers tackled the challenge of accurately estimating missing traffic data in intelligent transportation systems (ITS). They proposed a novel approach called Multi-BiSTGN, a multi-view bidirectional spatiotemporal graph network. This method comprehensively described traffic conditions from different temporal correlation views and fused them to impute missing data. The model was trained using a novel loss function to optimize its parameters, and it was tested on real-world traffic datasets from Wuhan, China. Results indicated that Multi-BiSTGN outperformed ten existing methods across various missing data types and rates, demonstrating its effectiveness in capturing nonlinear spatiotemporal correlations of missing traffic flow patterns.

Wang-2022-**Urban traffic flow prediction a dynamic temporal graph network considering missing values** ❤️

In this study, the researchers addressed the challenges of accurate traffic flow prediction in Intelligent Transportation Systems (ITS), particularly dealing with missing values and dynamic spatial relationships in traffic flow data. They introduced a dynamic temporal graph neural network (D-TGNM) that extended the Traffic BERT model to learn dynamic spatial associations, combined with a temporal graph neural network considering missing values (TGNM) to mine traffic flow patterns in missing data scenarios. The D-TGNM model, trained with a novel loss function, outperformed ten existing baselines in predicting traffic flow, showcasing its effectiveness under various missing data scenarios in an actual traffic dataset from Wuhan, China.

Ye-2021-**Traffic Data Imputation with Ensemble Convolutional Autoencoder**  🚩

In this study, the researchers aimed to improve traffic data imputation for Intelligent Transportation Systems by addressing the issue of incomplete data. They introduced an ensemble model called the ensemble convolutional autoencoder, which reconstructed observed and missing values into a two-dimensional matrix, leveraging spatial-temporal relations. This model, featuring convolutional and deconvolutional layers and trained with different input feature maps, outperformed other imputation methods. The experimental results demonstrated its ability to achieve higher accuracy and stable performance across various missing data scenarios with different types and rates, enhancing the quality of traffic data for related applications.

Yu-2020-**Forecasting road traffic speeds by considering area-wide spatio-temporal dependencies based on a graph convolutional neural network (GCN)** ❤️

In this study, researchers aimed to improve traffic forecasting in urban transportation networks by incorporating both temporal and spatial correlations. They developed a novel graph-based neural network that extended the existing graph convolutional neural network (GCN), allowing for differentiation in the intensity of connecting to neighbor roads. This model outperformed the original GCN model in forecasting traffic states and revealed hidden aspects of real traffic propagation through estimated adjacency matrices. Additionally, a generative adversarial framework was employed to enhance the realism of forecasted traffic states, contributing to more accurate traffic predictions.

Yuan-2023-**STGAN- Spatio-Temporal Generative Adversarial Network for Traffic Data Imputation** ❤️
In this study, the researchers aimed to address the challenge of efficiently imputing traffic data corrupted by noise and missing entries in Intelligent Transportation Systems (ITS). They introduced a novel spatio-temporal Generative Adversarial Network (GAN) model called STGAN. This model incorporated generative and center loss functions to minimize reconstruction errors and ensure local spatio-temporal distribution conformity. The discriminator used a convolutional neural network to assess global spatio-temporal distribution. The generator's network architecture included skip-connections to preserve well-preserved data and dilated convolution to capture spatio-temporal correlations. Experimental results demonstrated that STGAN outperformed other competitive traffic data imputation methods, improving the performance of ITS applications like congestion prediction and route guidance.

Zhang-2022-**Missing Data Repairs for Traffic Flow With Self-Attention Generative Adversarial Imputation Net** ❤️
In this study, the researchers addressed the challenge of missing data in large-scale sensor-collected time series data, particularly in the context of traffic flow data. They proposed a model called SA-GAIN (Self-Attention Generative Adversarial Imputation Net) that combined a self-attention mechanism, an auto-encoder, and a generative adversarial network. The self-attention mechanism helped capture correlations between spatially-distributed sensors at different time points, and adversarial training improved imputation accuracy. Experimental results demonstrated that SA-GAIN outperformed other imputation models in effectively filling missing data in traffic flow datasets.

Zhu-2022-**A Higher-Order Motif-Based Spatiotemporal Graph Imputation Approach for Transportation Networks** ❤️

In this research, the scientists addressed the issue of missing traffic data caused by incomplete coverage and data collector failures during data collection. They proposed a spatiotemporal imputation approach that leveraged motif-based graph aggregation. This approach utilized motif discovery and a graph convolution network (GCN) to aggregate correlated segment attributes of missing data segments. Additionally, a multitime dimension imputation model based on bidirectional long short-term memory (Bi-LSTM) incorporated various temporal dependencies. Experimental results using real-world datasets demonstrated the feasibility and accuracy of the proposed approach for addressing both random and continuous data missing in traffic data.

Zhuang-2018 -**Innovative method for traffic data imputation based on convolutional neural network** ❤️

In this study, the researchers aimed to address the challenge of missing traffic data by proposing an innovative imputation method. They transformed raw traffic data into spatial-temporal images and applied a convolutional neural network (CNN)-based context encoder to reconstruct complete images from the missing source. Their experiments, conducted on three months of data from 256 loop detectors, demonstrated that this novel approach significantly improved imputation accuracy compared to two state-of-the-art methods, offering a stable error distribution.







This repo contains important papers related to the `Temporal Missing Data imputation`


## Authors

- [shriyanksomvanshi](https://github.com/shriyanksomvanshi)

