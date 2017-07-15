# Quora Question Pair competition

Data science competition on Kaggle

## Links

+ Competition \
https://www.kaggle.com/c/quora-question-pairs

+ Name Entity Recognition: (silver medal kernel) \
https://www.kaggle.com/hubert0527/spacy-name-entity-recognition

## Reports

+ Report (written in mark down) \
https://hackmd.io/AwYw7ArAjGUEYFoDMBDMIEBYAmKkIE5hgDlMoDtMkA2AJjmBSA==

+ PDF report \
https://github.com/GGNLP/QuoraQuestionPair-competition/blob/master/Kaggle-Competition-Quora-Question-Pairs.pdf

+ Hackmd Slides \
https://hackmd.io/p/Hk7IYhoMZ#/



## Representation video

https://www.youtube.com/watch?v=IwW2dmrZLjI

## Important kernels:

+ Main kernel -- [[Link]](https://github.com/GGNLP/QuoraQuestionPair-competition/blob/master/HubertLin/Model_RNN_with_leaks.ipynb)
> We found several training and data problems and improve most of them in this version. It generates our best single model.

+ Xgboost -- [[Link]](https://github.com/GGNLP/QuoraQuestionPair-competition/blob/master/HubertLin/Model_Xgboost.ipynb)
> Utilize the highly optimized GBDT library Xgboost to do training. It generates out second best single model.

+ Blending -- [[Link]](https://github.com/GGNLP/QuoraQuestionPair-competition/blob/master/HubertLin/Model_blending.ipynb)

> Since we made some mistake in the stacking stage, I turned out a new way to stack(blend) the models.
> This method works better on string classifier, which is one of its benefit.

+ Name Entity Replacement -- [[Link]](https://github.com/GGNLP/QuoraQuestionPair-competition/blob/master/dataset/release_kernel/Name_Entity_Recognition.ipynb)

> We released this kernel on Kaggle and won a silver medal from the community. -- [[Link]](https://www.kaggle.com/hubert0527/spacy-name-entity-recognition) \
> This kernel utilize the great NLP library SpaCy to do name entity recognition task, then replace the recognized entity with its enitity name in the question. It highly speed up the training time.
