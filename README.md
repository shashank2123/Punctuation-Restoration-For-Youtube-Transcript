# Punctuation-Restoration-For-Youtube-Transcript
Though people rarely speak in complete sentences, punctuation confers many benefits to the readers of transcribed speech. Unfortunately, most ASR systems do not produce punctuated output. So, Here We are going to build Punctuation Restoration System for youtube transcript just by providing  URL of Youtube Video.

### Installation
```
pip install -r requirements.txt
```

### Run
```
python punctuators.py https://youtu.be/UFDOY1wOOz0
```


### Dataset


We are using [TED – Ultimate Dataset](https://www.kaggle.com/miguelcorraljr/ted-ultimate-dataset) datasets.TED is devoted to spreading powerful ideas in just about any topic. These datasets contain over 4,000 TED talks including transcripts in many languages. For this task we are only using english language dataset. 

### Model
[`t5-base` ](https://huggingface.co/t5-base) is fine-tuned with [TED – Ultimate Dataset](https://www.kaggle.com/miguelcorraljr/ted-ultimate-dataset) dataset.
Completly trained model is [here](https://huggingface.co/shashank2123/t5-base-fine-tuned-for-Punctuation-Restoration)

### Evaluation Metrics 
#### How Precisions , Recall , F1-Score are Calculated?

We used ROUGE metrics  technique but only for punctuation marks. Rouge metrics uses all the tokens for comparision. In this metrics evaluation which extractes punctuation marks and its both adjacent word tokens, then Precisions,Recall and F1-Score are calculated.

For example:-

```

candidate sentence = "Hello How are you?"

reference setence = "Hello, How are you?"

cand_grams = ["you ? </s>"]

ref_grams = ["Hello , How" , "you ? </s>" ]

common_grams = ["you ? </s>"]

```

**Recall**

The recall counts the number of overlapping n-grams found in both the model output and reference — then divides this number by the total number of n-grams in the reference. It looks like this: 

<center><img src = "https://miro.medium.com/max/3000/1*XEnhQJxKbEySimh1PPWPnQ.png" height = 300 width = 600 ></center>

For above example recall would be **0.5** that is number of `common_grams` dividing by total number of `ref_grams`.

**Precision**

precision metric — which is calculated in almost the exact same way, but rather than dividing by the reference n-gram count, we divide by the model n-gram count.

<center><img src = "https://miro.medium.com/max/3000/1*aSd89F6kupr3znW71Qmb3Q.png" height = 100 width = 600 ></center>

For above example precision would be **1** that is number of `common_grams` dividing by total number of `cand_grams`.

**F1-Score**

Now that we have both the recall and precision values, we can use them to calculate our F1 score like so :-

<center><img src = "https://miro.medium.com/max/875/1*zYuwaCDNpYf51H5S4DpDRA.png" height = 100 width = 300 ></center>

For above example F1-score would be **0.666**

### Result
| Dataset       |  Precision | Recall  |F1-Score |
|:-------------:|:----------:|:-------:|:-------:|
| Validation    | 0.896      | 0.793   | 0.838   |
| Test          | 0.887      | 0.752   | 0.814   |

|Punctuation marks| Accuracy      |
|:---------------:|:-------------:|
| "."             | 0.863         | 
| ","             | 0.683         |  
| "?"             | 0.769         | 
| ":"             | 0.708         |
