# NLP: Hidden Markov Model (HMM) POS Tagger and Brill Tagger

In this project, our aim is to tune, compare, and contrast the performance of the Hidden Markov Model (HMM) POS tagger and the Brill POS tagger.
To perform this task, we will train these two taggers using data from a specific domain and test their accuracy in predicting tag sequences from data belonging to the same domain and data from a different domain.


## 3-rd Party Libraries
Apart from `nltk` and `pandas`, we have used the following 3rd party libraries:

* `main.py L:355` used **`scikit-learn`** for splitting the dataset into train set and cross validation set.


## How to Execute?

To run this project,

1. Download the repository as a zip file.
2. Extract the zip to get the project folder.
3. Open Terminal in the directory you extracted the project folder to. 
4. Change directory to the project folder using:

    `cd part-of-speech-taggers-main`
5. Install the required libraries, **NLTK** and **scikit-learn** using the following commands:

    `pip3 install nltk`

    `pip3 install -U scikit-learn`
 
6. Now to execute the code, use any of the following commands (in the current directory):

**HMM Tagger Predictions:**
`python3 src/main.py --tagger hmm --train data/train.txt --test data/test.txt --output output/test_hmm.txt`

**Brill Tagger Predictions:**
`python3 src/main.py --tagger brill --train data/train.txt --test data/test.txt --output output/test_brill.txt`    




## Description of the execution command

Our program **src/main.py** that takes four command-line options. The first is **--tagger** to indicate the tagger type, second is **--train** for the path to a training corpus, the third option is **--test** for the path to a test corpus, and the fourth option is **--output** for the output file.

The two possible values for --tagger option are: 

* `hmm` for the Hidden Markov Model POS Tagger

* `brill` for the Brill POS Tagger

The assignment's training data can be found in [data/train.txt](data/train.txt), the in-domain test data can be found in [data/test.txt](data/test.txt), and the out-of-domain test data can be found in [data/test_ood.txt](data/test_ood.txt). 

The output file must be generated in the [output/](output/) directory.

So specifying these paths, one example of a possible execution command is:

`python3 src/main.py --tagger hmm --train data/train.txt --test data/test.txt --output output/test_hmm.txt`



## References

https://docs.huihoo.com/nltk/0.9.5/api/nltk.tag.hmm.HiddenMarkovModelTrainer-class.html

https://tedboy.github.io/nlps/generated/generated/nltk.tag.HiddenMarkovModelTagger.html

https://www.kite.com/python/docs/nltk.HiddenMarkovModelTagger.train

https://gist.github.com/blumonkey/007955ec2f67119e0909

https://docs.huihoo.com/nltk/0.9.5/api/nltk.tag.brill-module.html

https://www.nltk.org/api/nltk.tag.brill_trainer.html

https://www.nltk.org/_modules/nltk/tag/brill.html

https://www.geeksforgeeks.org/nlp-brill-tagger/

https://www.nltk.org/howto/probability.html
