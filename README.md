# Industrial-Ball-Bearing-Failure-Prediction
This project focuses on the application of statistical time-domain features for the prediction of ball bearing failures. Some of the standard machine learning techniques are applied on publicly available datasets from Case Western Reserve University (CWRU) to predict the failures.

**Desired Outcome:** Being able to predict if ball bearings would fail.

# **Approach:**

<img width="545" height="267" alt="image" src="https://github.com/user-attachments/assets/02c7eabb-3134-41ad-835b-513dc8f3314f" />

# **Results:**

Here is a summary of the results from my project:

<img width="1322" height="428" alt="image" src="https://github.com/user-attachments/assets/10e82e11-c61d-46c1-a29f-12c4b7576996" />

**1) Linear Regression Results:**

<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/2b8a7cc1-878e-4a82-9f87-5eabc78aa1fc" />
<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/ca4c23dd-4ab4-4dc5-94ae-4752e6b709ee" />

**2) Random Forest Results:**

<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/968087e5-3862-4c80-b629-c59f40a1ad0a" />
<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/607ef924-9b2a-4ea9-ac03-5a86904ae561" />

**3) Support Vector Machine(SVM) Results:**

<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/e634ed9d-d150-4ef9-a0e8-dc1d52d46383" />
<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/ab275349-4fa2-464a-a25d-6b0b2d847327" />

**4) K-Nearest Neighbors(KNN) Results:**

<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/6ce77579-5ca5-4d7c-8257-d8be0ac5643f" />
<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/2ada6761-881f-4a38-9091-a0d5117cf8c2" />

**5) Gradient Boosting Results:**

<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/5412d06c-0de6-434c-80b5-8ca00d7f6884" />
<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/b3bb978b-94b4-445d-bed5-220b3ac1a3bd" />

**6) Convolutional Neural Networks(CNN) Results:**

<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/0e9c9a83-73ca-4efb-871e-a263b58250fe" />
<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/6d97b9f3-4068-4317-860f-518d9adc0c9f" />

**7) Long Short-Term Memory(LSTM) Results:**

<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/170dfbae-f786-4e50-b5a4-f5bbc343c836" />
<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/d51ea032-dbcb-4043-b61f-88b890ac6315" />

# **6. Results and Conclusion**
Overall, the models were very successful in predicting both faulty and non faulty ball bearings. All of my traditional machine learning models have an accuracy of 97 percent, with Logistic Regression being the exception at a still high 94 percent whereas the deep learning models both have an accuracy of 96 percent. Despite the Deep Learning Models having an AUC score higher than most of the traditional models, Overall traditional models were more successful. This is likely due to the limited data size, absence of raw signal sequences, and shallow network layers. This is because deep models thrive on large amounts of training data, temporal/spatial patterns, and more layers/tuning to optimize their performance.

## **6.1 Traditional Model Conclusions**
Logistic Regression, SVM, Random Forest, KNN, and Gradient Boosting are all commonly used for classifi cation tasks. Their performance depends heavily on the dataset characteristics. In this specifi c case, the accuracy scores and confusion matrices provide insights into their relative performance.

**Accuracy:** Higher accuracy suggests a better predictive ability. Looking at the results table we can see that all of the traditional machine learning models have a high accuracy, meaning they all likely have great predictive ability. Looking at that table we can also see that Logistic Regression has an accuracy lower than even the deep learning models. This is likely due to how Logistic Regression handles the datasets compared to the other models. Logistic regression works best on simple, linearly separable datasets. Its lower accuracy relative to the other models suggests that the data is not simple or it is not linearly separable, perhaps even having complex interactions between features. The other models having a higher accuracy also imply these facts about my dataset as they are better at handling complex data that isn’t necessarily linearly separable.

**True positive rate and false positive rate:** The true positive rate mirrors the accuracy both in their relative values and what they tell us about the model's performance against my dataset. The false positive rate similarly mirrors accuracy and the relationship between the models but it is inversed, with Logistic Regression having the largest false positive rate.

**AUC scores:** My AUC scores for the traditional models have a slightly different distribution than my previously discussed metrics. KNNs have the lowest AUC score, being the only traditional model along with Logistic Regression and Gradient Boosting(barely) to have a lower AUC score. This is likely because KNN doesn’t produce true probability scores but rather counts the amount of occurrences in a class among neighbors, which limits its ability to rank its predictions, which is what AUC measures. Logistic Regression has a lower score likely due to its inability to assign well-separated probability scores for datasets that aren’t linearly separable, once again pointing to the fact that this dataset isn’t completely linearly separable. The other models have higher AUCs due to their abilities to separate classes well and have reliable confi dence estimates of the output probabilities. Looking at the ROC curves also back what the AUC scores tell us as KNN and Logistic Regression are less steep before fl attening out compared to the other models, mirroring their lower AUC score. In general the AUC scores are all very high refl ecting that all of the models rank predictions well.

## **6.2 Deep Learning Model Conclusions**
Deep learning models often require very large amounts of data and are typically used for complex pattern recognition including image recognition or natural language processing. Their lower accuracy and above average AUC score tell us that in general while these models are better at separating classes they aren’t as good at the specific threshold used for accuracy.

**Accuracy:** CNN and LSTM’s lower accuracy compared to the traditional models suggest that there wasn’t as much data as these models would have liked to train on. This is compounded with the fact that I did extensive feature engineering before training these models, leading to them attempting to directly learn features that have already been manually extracted.

**True positive rate and false positive rate:** As with my traditional learning models these mirror accuracy in how they are distributed, inversely in the case of the false positive rate. This is because my models make few gross misclassifi cations, leading to a high true positive rate, low false positive rate, and high accuracy, which makes these values appear to mirror each other.

**AUC scores:** The Deep Learning Models above average AUC relative to the traditional models contrasts with its accuracy in relation to the same models. This difference comes from the Deep Learning Models learning complex patterns better, leading to them often assigning higher scores to true positives and lower scores to negatives which improves ranking quality leading to a better AUC. Another reason is that since AUC disregards the threshold that accuracy focuses on it leads to a higher AUC. Looking at the ROC curves for my Deep Learning Models we see that they are steeper than the models that have a lower AUC, mirroring these models above average AUC.
