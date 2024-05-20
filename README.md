# Algorithmic Fairness Mandatory Assignments Spring 2024
## Group Equalised Odds

***

### Mandatory assignment 1

- Task 1 (Classifiers and fairness considerations)
- Task 2 (Explaining white-box models)
- Task 3 (Model-agnostic explanations)
- Task 4 (Reflection)


Requirements: [requirements.txt](https://github.com/Xannadoo/Algorithmic_Fairness_Mandatories/blob/main/Mandatory_1/requirements.txt)

This work has been tested to work with the following:

> Python 3.11.7
>
> folktables==0.0.12 <br>
> numpy==1.26.3 <br>
> pandas==2.2.0 <br>
> scikit-learn==1.4.1.post1 <br>
> matplotlib==3.8.2 <br>
> seaborn==0.13.2 <br>
> shap==0.44.1

***

### Mandatory assignment 2

- Task: Train 3 classifiers
    + Raw data
    + De-correlated data
    + FairPCA data
 
Requirements: [requirements.txt](https://github.com/Xannadoo/Algorithmic_Fairness_Mandatories/blob/main/Mandatory_2/requirements.txt)

This work has been tested to work with the following:

> Python 3.10.12
>
> folktables==0.0.12 <br>
> numpy==1.26.3 <br>
> pandas==2.2.0 <br>
> scikit-learn==1.4.1.post1 <br>
> matplotlib==3.8.2 <br>
> seaborn==0.13.2 <br>
> shap==0.44.1

***

### Exam Assignment

- Dataset: [Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)

- We will explore the dataset, for its intended use and other possible uses, by considering its statistics, features, correllations etc.
- We will consider attributes that should be protected for and attempt to debias the data set using geometric/fairPCA methods
- We will also consider just removing those attributes to create a naive model
- We will reflect upon the consquences of such a dataset, both for idealistic purposes (early indentification of student who may need extra support) and possible other uses (for example, preventing the enrollment of students who are likely to drop out based on non-academic features), taking into consideration that protected attributes form part of this dataset.

Requirements: [requirements.txt](https://github.com/Xannadoo/Algorithmic_Fairness_Mandatories/blob/main/Exam/requirements.txt)

This work has been tested to work with the following:

> Python 3.10.14
>
> matplotlib==3.8.4 <br>
> numpy==1.26.4 <br>
> pandas==2.2.2 <br>
> scikit-learn==1.4.2 <br>
> scipy==1.13.0 <br>
> seaborn==0.13.2 <br>
> shap==0.45.1 <br>
> tqdm==4.66.4 <br>
> ucimlrepo==0.0.6 <br>
> certifi==2024.2.2 <br>
