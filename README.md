# Author_Profiling
Multilingual Author Profiling project for the Learning From Data Project.

### How To Run:

1) To Run the scrips - keep the training and test directory under the same directory as the scrips are.

2) Specify the following command line arguments while you run the python scripis:

	python English_svm.py -tr "training/english/" -te "test/english/"

	python Dutch_svm.py -tr "training/dutch/" -te "test/dutch/"

	python Italian_svm.py -tr "training/italian/" -te "test/italian/"

	python Spanish_svm.py -tr "training/spanish/" -te "test/spanish/"


### Please Cite the paper if you use the code/data:

```ruby
@InProceedings{10.1007/978-3-030-51859-2_46,
author="Rahman, Md. Ataur
and Akter, Yeasmin Ara",
editor="Chen, Joy Iong-Zong
and Tavares, Jo{\~a}o Manuel R. S.
and Shakya, Subarna
and Iliyasu, Abdullah M.",
title="Multi-lingual Author Profiling: Predicting Gender and Age from Tweets!",
booktitle="Image Processing and Capsule Networks",
year="2021",
publisher="Springer International Publishing",
address="Cham",
pages="505--513",
abstract="This article describes how we build a multi-lingual classification system for author profiling. We have used Twitter corpus for English, Dutch, Italian and Spanish languages for building different models incorporating SVM classifier that predicts the gender and age of an author. We evaluated each model using 3-fold cross-validation on the training dataset for each of these languages. The overall maximum average accuracy for gender classification was 81.3{\%} for Spanish while for classification of age we achieved a maximum accuracy score of 70.3{\%} for English using the cross-validation scheme. For other languages, the results were between 64--76{\%}.",
isbn="978-3-030-51859-2"
}
```
