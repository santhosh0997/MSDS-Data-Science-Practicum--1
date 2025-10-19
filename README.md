# Marketplace Review Authenticity Filter: Human vs AI vs Human-Assisted (with Review Analytics)

## All data files for this project are available at:
https://drive.google.com/drive/folders/14C4BT7mpbZNT29ydpMsPIjrT7KcvzVlx?usp=sharing
### Accessing the code:
"complete_code_analysis.ipynb" is the complete analysis for the project.

To run the streamlit use:

streamlit run streamlitapp.py

## 1. Introduction
In a world where we are overwhelmed by artificial voices, the question is, are we able to trust what we read?
User reviews are the motivation behind online marketplaces like Amazon, Yelp, and Google Maps. Nevertheless, the increasing availability of AI models has allowed malicious actors to create fake reviews in a large number, undermining the credibility of the platform and consumer trust.

Why it matters:

●	The increased rate of the LLM proliferation has compounded the review authenticity challenge.

●	Moral uprightness requires sound strategies to fight miscommunication.

●	AI detection models require explainability to facilitate user trust and regulatory compliance.

The market research findings have further suggested that the content marketing market is likely to hit 17.6 billion dollars by 2032 with more than half being automated content.

This expanding environment requires the methodic response to divide machine-based persuasion and human articulation.

<img width="723" height="482" alt="image" src="https://github.com/user-attachments/assets/387cba49-84d0-4672-8b17-cce62ba0b762" />

## 2. Problem Statement

On the one hand, it may seem simple to detect the fake reviews. We have been sifting spam mail and spotting frauds long enough anyway. However, the emergence of big language models (LLMs) has changed the game and put it on a new level.

In contrast to classic spam, which can be detected with low grammar, clumsy wording, or a shady URL, AI generated reviews are written so as to look just like those written by a normal person. They are literate, logical and contextual. Indeed, they usually seem to be more refined than a real review, written by an average user.

It is the irony that, the same things that make AI text helpful in clarity, fluency and consistency make it harmful when applied improperly in internet markets.
Finding them is not that hard, but interpretable detection models need to demonstrate the justification of their classifications in a transparent manner and be able to be ethically used in marketplaces.

## 3. Objectives

1.	Build a machine-learning system that will differentiate between human and AI-generated reviews.
2.	Embedding and linguistic features are better to be used to improve accuracy.
3.	Make sure that it is explainable using interpretable model outputs.
4.	Compare the classical and deep-learning models to determine which works the best.

## 4.Literature Review

The efforts to address the issue of AI-generated reviews should be based on the larger literature on synthetic text detection. In the last few years, academics and professionals have come to consider how machine learning can create as well as detect artificial content. This section gives an overview of the most impactful work, its strong and weak sides, and its applicability to the current project.


#### 4.1 Preliminent Work in Neural Fake Text Detection.
Zellers et al. (2019) also conducted one of the pioneering studies in this area, presenting Grover, a model to produce and identify neural fake news. The anomaly in their work was paradoxical: to recognize AI-specific text, it is frequently better to utilize AI itself. Not only was Grover told to generate synthetic news but also learn to recognize it, and it can perform well in the task of distinguishing between machine-written and human-written items.

This dual-use strategy highlighted a very important theme: the development of the detection tools should go in the same pace as the generation tools. The detectors should be sophisticated as the models become advanced.

#### 4.2 AI Content Ethical Frameworks.
About the same period, Solaiman et al. (2019) at OpenAI went to look into the ethical concerns of releasing powerful language models. Their study highlighted how the uncontrolled access to LLMs would make them be abused, such as the generation of fake reviews, propaganda, and misinformation in large volumes. They suggested planned disclosure methods and emphasized on responsible disclosure.

The work is specifically applicable since it does not only present the detection problem as a technical one but also as an ethical one. Platforms need to strike a balance between innovations and controls against the manipulation of the users.

#### 4.3 Statistical and Embedding-Based Method.
Leaving ethics aside, Jawahar et al. (2020) explored useful detection techniques. They also conducted comparisons between both transformer-based and statistical representations (word frequency distributions, perplexity scores, etc) in their study. They discovered that embeddings do miss semantic subtlety, yet statistical characteristics tend to highlight stylistic aberrations which embeddings are ignorant of.

This discovery paved the way to hybrid strategies, which unite the merits of the two type of features a strategy central in the current project.

#### 4.4 Explainability and Visualization Tools.
Gehrmann, Strobelt, and Rush (2019) were another force that contributed to the development of the Giant Language Model Test Room (GLTR). GLTR plots the predictability of every word in a piece of writing, and thus humans could easily detect machine writing. Considering the example, AI models tend to select the most likely words, which results in unnaturally likely sequences.
The relevance of GLTR is that it is explainable. It allows users to understand why a piece of text may be synthetic, as opposed to detection being a black box. This is quite consistent with the objectives of this project whose focus is more on transparency as well as on accuracy.


#### 4.5 Positioning This Project
Based on these premises, the current publication incorporates:

●	The two sides of Grover (AI to detect AI).

●	The Solaiman et al. ethical framing.

●	The Jawahar et al. hybrid feature approach.

●	The explainability culture of GLTR.

Through the combination of these strands, the project places itself in the crossroad of technical rigor, ethical responsibility and practical applicability.
Such researches point to the need of hybrid and transparent systems that is exactly what this project deals with.


## 5. Methodology Overview

<img width="1213" height="220" alt="image" src="https://github.com/user-attachments/assets/112400b1-d4ca-4036-9312-4703d3db2cae" />



The methodology is based on a traditional data science lifecycle, modified to the specifics of the text authenticity detection:
●	Data collection - obtaining human-made and AI-generated reviews in various sources.

●	Preprocessing - cleansing, normalizing and pre-preparation of text to be analyzed.

●	Feature Engineering - the extraction of linguistic features and embedding-based features.

●	Model Development - training and comparison of the classical ML and deep-learning models.

●	Assessment - measuring accuracy, precision, recall and F1-scores.

●	Explainability - the use of interpretability techniques to bring about transparency.

All the stages are not solitary but repetitive. The knowledge of evaluation is advised to evaluation-based feature engineering and explainability to preprocessing and model refinements.


## 6. Data Collection and Setup
Sources:

●	Consumer Opinion Reviews: Yelp and Amazon.

●	HC3 and Kaggle datasets of benchmark AI-text samples.

●	Serp API reviews on Google Maps to get diversity on the ground.


#### Curation:
An equal number of AI-generated and human reviews.
The labeling and similar formatting were standardized, which guaranteed model reliability.

In brief, the information gathering and preparation stage was approximately a laying of a strong foundation. The sourcing, balancing and curation of reviews made the project make sure that the models will be trained on the data that is a realistic representation of the real world.
In the absence of this work, even the most advanced algorithms would have been working on sand. It was now prepared to proceed to the second step, which is data cleaning and preprocessing.


## 7. Data Cleaning and Preprocessing

When any machine learning project is based on the foundation of the data collection, then the scaffold that gives everything its shape is the data cleaning and preprocessing. Raw text as it is delivered by Internet websites is noisy, inconsistent and filled with quirks. Any model needs to be standardized and deprived of artifacts that do not matter before learning any meaningful patterns.

Preprocessing is even more important in the framework of the detection of AI-generated reviews. Authenticity detection, in contrast to the broad patterns that can do with general tasks like sentiment analysis, needs the detector to be sensitive to subtle stylistic clues. The presence of an odd burst of repetition, the presence of a misplaced comma, or the presence of an excessively homogenous sentence structure will all be indicators of synthetic origin. Having not been processed carefully, such signals are in danger of being lost in noise.

The raw text was preprocessed in a number of ways:

●	The use of tokens and small letters.

●	Elimination of stopwords, punctuations, and unnecessary spaces.

●	Dealing with unusual/ misspelled words.

●	Mitigation of bias by normalizing text length.

Such a standardized corpus facilitated the use of reliable linguistic and semantic features.

<img width="387" height="354" alt="image" src="https://github.com/user-attachments/assets/b3ac0fa9-55bc-4915-b414-3d50d297c9fc" />


## 8. Feature Engineering

In case preprocessing removes the dirt on the canvas, feature engineering fills the picture. Features in machine learning represent the form of data used by a model to make predictions. In the case where text authenticity detection is used to get the correct information, the issue comes in the need to identify the superficial strangeness of writing and the semantic content of words.

In this project, the hybrid feature engineering approach is used, which is a combination of linguistic features and embedding-based representations. 
The reasoning is not complicated: linguistic characteristics forces the way something is written and embeddings forces the sayings. Collectively, they will offer a multi-dimensional prism through which we can differentiate between human-generated and AI-generated reviews.

#### A. Linguistic Features
 These capture surface-level and syntactic properties of writing:
 
●	These represent shallow and structural features of writing.

●	Perplexity & Entropy: calculate unpredictability and fluency.

●	Burstiness: picks up unnatural repetition.

●	Readability Scores: evaluate grammatical coherence.

●	Stopword Frequency & Repetition of Rare Words: find artificial structure.

#### Exploratory Data Analysis:

<img width="550" height="429" alt="image" src="https://github.com/user-attachments/assets/8813e973-5046-4d64-aaf3-2d77574e3012" />

<img width="524" height="367" alt="image" src="https://github.com/user-attachments/assets/ae261b16-c0e1-40b5-9981-dfac4a99833a" />



#### B. Vectorization & Embeddings

●	TF-IDF Vectorization: measures the significance of terms.

●	BERT Embeddings: learn contextual semantics with trained language models.

These constitute the hybrid feature space that drives model training.

## 9. Model Development

Algorithms Evaluated:

●	Logistic Regression

●	Support Vector Machine

●	Random Forest

●	XGBoost

●	BERT Transformer (Classifier head)

Each of the model was trained like this:

Setting	Description	Observations

TF-IDF Only	Classical ML models + TF-IDF features	Accuracy ≈ 0.5 – no meaningful learning

Embeddings Only	DistilBERT embeddings without linguistic features	Accuracy ≈ 0.5 – poor discrimination

Hybrid Model	Linguistic + Embeddings	Significant performance boost

<img width="806" height="415" alt="image" src="https://github.com/user-attachments/assets/d1ce374b-7e9d-4340-87bc-fa049a54a434" />

## 10. Evaluation and Results

<img width="930" height="312" alt="image" src="https://github.com/user-attachments/assets/3611c7bb-e11b-4485-b68d-b219d8a7f3ca" />

<img width="799" height="382" alt="image" src="https://github.com/user-attachments/assets/78e3a8ea-f2bf-41f8-b5e5-ea73d7494ff4" />

The hybrid system with both linguistic and embedding characteristic performed better than the others.

Notably:

●	Better generalization was done with Logistic Regression and XGBoost.

●	F1-scores, accuracy and precision all were more than 90%.

●	The hybrid strategy along with performance improvements brought interpretability.


## 11. Explainability and Visualization
The interpretations of the model  were made with the help of SHAP visualizations.

They show why some of the reviews are classified as AI-generated:
●	Extra sentence constructions.

●	Repetitive phrasing

●	Uniform sentiment tones

The insights increase user trust and enable auditors to track system activity.

<img width="506" height="442" alt="image" src="https://github.com/user-attachments/assets/b71f311e-f50b-4da6-9e28-e2b0e172dde8" />

<img width="684" height="286" alt="image" src="https://github.com/user-attachments/assets/f11e04ba-4873-4ec9-9b92-772cb1b2893b" />

## 12. Discussion
The results highlights the fact that that hybrid architectures provide a higher performance in terms of text authenticity detection. Whereas embeddings are sensitive to the semantic changes, also sensitive to the irregularities of style, between them.

Challenges include:

●	Neural artificially intelligent LLMs with human burstiness.

●	Limitations of datasets in languages and styles of culture.

●	The process of retraining models constantly because AI writing is adopted.

## 13. Future Work

Future extensions aim to:

●	Increase the size of the corpus through multilingual data.

●	Introduce live moderation APIs of the marketplace.

●	Add human readable explainable dashboards.

●	Discover transformer hyper-optimization in domain text (e.g., e-commerce reviews).

## 14. Conclusion
This project highlights the necessity of AI detection systems in online market.

The hybrid model combines linguistic features with embeddings to produce better result, that is, the platforms will be transparent and the user will trust the reviews he/she is reading.

## 15. References 
1.	Zellers, R., Holtzman, A., Rashkin, H., Bisk, Y., Farhadi, A., Roesner, F., & Choi, Y. (2019). Defending Against Neural Fake News (Grover). arXiv:1905.12616. https://arxiv.org/abs/1905.12616
2.	 Solaiman, I., et al. (2019). Release Strategies and the Social Impacts of Language Models. arXiv:1908.09203. https://arxiv.org/abs/1908.09203
3.	 Jawahar, G., Abdul-Mageed, M., & Laks, V. (2020). Automatic Detection of AI-Generated Text. arXiv:2009.11474. https://arxiv.org/abs/2009.11474
4.	Gehrmann, S., Strobelt, H., & Rush, A. M. (2019). GLTR: Statistical Detection and Visualization of Generated Text. arXiv:1906.04043. https://arxiv.org/abs/1906.04043




