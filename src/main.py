# This program receives the tagger type and the path to a test file
# as command line parameters and outputs the POS tagged version of that file.

import argparse

from sklearn.model_selection import train_test_split

import nltk
nltk.download('punkt')
from nltk.tag import RegexpTagger, BrillTaggerTrainer
from nltk.tbl.template import Template
from nltk.tag.brill import Pos, Word, brill24, fntbl37, nltkdemo18, nltkdemo18plus
from nltk.tag import hmm
from nltk.probability import LidstoneProbDist, MLEProbDist, LaplaceProbDist, ELEProbDist, WittenBellProbDist



# Initialize the argument parser
parser = argparse.ArgumentParser()

# Add the parameters we will pass from cli
parser.add_argument('--tagger',help='tagger type')
parser.add_argument('--train',help='path to the input training data file')
parser.add_argument('--test',help='path to the input test data file')
parser.add_argument('--output',help='path to the output file')

# Parse the arguments
args = parser.parse_args() 
#print(args)

# Name of tagger_type
tagger_type = args.tagger
# Path to input training data file
input_train_data_path= args.train
# Path to input test data file
input_test_data_path= args.test
# Path to output file
output_file_path= args.output







'''

Function to get data from the specified file path, do preprocessing, get in format of what NLTK expects for its functions

'''

def pre_process_data(file_path):

    with open(file_path, 'r') as input_file:


            #This will create a 1D list of words (with whitespaces removed from end)
            cleaned_lines= []
            for line in input_file:
                cleaned_lines.append(line.rstrip())
 

            # Now we will convert this 1D list of words to 2D list of sentences
            list_of_sentences=[]            
            sentence=[]
            for word in cleaned_lines:
                if(word==""):
                    list_of_sentences.append(sentence)
                    #print(sentence)
                    sentence=[]
                else:
                    sentence.append(word)
            
            #add last sentence to list of sentences ( won't be added by above loop as it does not end with "" )
            list_of_sentences.append(sentence)
            

            #We need to convert each word in each sentence as a tuple of (word, postag) because that is what the nltk tagger takes as input
            for i in range(len(list_of_sentences)):
                
                for j in range(len(list_of_sentences[i])):

                    list_of_sentences[i][j]= tuple((list_of_sentences[i][j].split(" ")))
                    
            
            
            return list_of_sentences






'''

Function to calcualate accuracy by comparing predicted tags of each word with truth tags of each word

'''

def calculateAccuracy(predictedTagswithWords,truthTagsWithWords):

    count=0 #will track no of correctly classified tags
    total_tags=0 #will track total no of tags in the data we are evaluting


   
    for i in range(len(truthTagsWithWords)):

        truth_sentence = truthTagsWithWords[i]
        
        predicted_sentence = predictedTagswithWords[i]
        
        #get predicted tag for each word in each sentence and then compare with corresponding truth tag of the word
        for j in range(len(truth_sentence)):

            # each word is a tuple of the form (word,tag) so tag is stored at position 1 in tuple
            truth_tag= truth_sentence[j][1] #get truth tag of word

            predicted_tag =  predicted_sentence[j][1] #get predicted tag of word

            if(truth_tag==predicted_tag):
                count= count+1
                #print(truth_tag+"============"+predicted_tag) #correctly predicted tag

            #else:
                #print(truth_tag+"=====!!!!!======="+predicted_tag) #incorrectly predicted tag

            total_tags = total_tags+1

    


    accuracy = count/ total_tags

    return accuracy










'''

Function to get predictions on cross val data and then call the function that will calculate accuracy

This function will be used while we train multiple models turing hyper-parameter tuning and then find accuracy of each model on cross val data.

'''

def evaluate_model(tagger, cross_val_set):


    cv_sentences_with_tag_removed = list(cross_val_set)
    
    #remove tags before prediction
    X_cv=[]
    for i in cv_sentences_with_tag_removed:
        l= []        
        for j in i:
            l.append(j[0])
        X_cv.append(l)    
                    

    predictedTags = [] #to store predictions for sentences with only words without tags

    #We will go through each sentence (untagged) in cross val set
    for sentence in X_cv:
        # get the prediction of tags for the sentence
        tags = tagger.tag(sentence)    
        predictedTags.append(tags)
            

    accuracy = calculateAccuracy(predictedTags, cross_val_set)
    return accuracy










'''

Function to train an HMM tagger using the passed arguments as the hyper-parameters and then return the model

'''


def train_hmm_tagger(train_set, estimator_hmm=None):


    #Training the model on train set ( word + pos tag for each word of the sentence)

    # Construct a HMMTrainer using the specified arguments
    hmm_model = hmm.HiddenMarkovModelTrainer()
    # Train on train_set
    hmm_tagger= hmm_model.train_supervised(train_set, estimator=estimator_hmm)
    
    return hmm_tagger







'''

Function to train a brill tagger using the passed arguments as the hyper-parameters and then return the model

'''


def train_brill_tagger(train_set, template_arg=None, max_rules_arg=10): 

    #Training the model on train set ( word + pos tag for each word of the sentence)

    # Initialize a basic set of rules (from the documentation for BrillTaggerTrainer)
    baseline = RegexpTagger([
    (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),   # cardinal numbers
    (r'(The|the|A|a|An|an)$', 'AT'),   # articles
    (r'.*able$', 'JJ'),                # adjectives
    (r'.*ness$', 'NN'),                # nouns formed from adjectives
    (r'.*ly$', 'RB'),                  # adverbs
    (r'.*s$', 'NNS'),                  # plural nouns
    (r'.*ing$', 'VBG'),                # gerunds
    (r'.*ed$', 'VBD'),                 # past tense verbs
    (r'.*', 'NN')                      # nouns (default)
    ])
    
    # reset the templates from previous runs
    Template._cleartemplates() 

    # Construct a BrillTaggerTrainer using the specified arguments
    brill_tagger_model = BrillTaggerTrainer(baseline, templates=template_arg, trace=False) 
    # Train on train_set
    brill_tagger = brill_tagger_model.train(train_set, max_rules=max_rules_arg)
    
    return brill_tagger


    #Reference: https://www.nltk.org/api/nltk.tag.brill_trainer.html
    #Reference: https://www.nltk.org/_modules/nltk/tag/brill.html
    #Reference: https://www.geeksforgeeks.org/nlp-brill-tagger/







'''

Function to get final predictions on the test files and then create the output file according to requirements

'''


def get_predictions_on_test_data(tagger):

    try:

        
            # Read input from "test.txt/ test_ood.txt" data file, call preprocess function we have defined above
            testing_sentences= pre_process_data(input_test_data_path)



            #Get only words without tags from testing_sentences as we are going to predict the tags for these sentences
            X_test=[]
            for i in testing_sentences:
                l= []        
                for j in i:
                    l.append(j[0])
                X_test.append(l)    




            #Get predictions on sentences with only words without tags
            list_of_all_predictions = [] #list to store predicted tags with new lines in between

            predicted_tags = [] #list to store all predicted tags

            #Go through each sentence (untagged) in test set and get prediction of tags for it
            for sentence in X_test:

                # get the prediction of tags for the sentence
                tags = tagger.tag(sentence)
                predicted_tags.append(tags)

                # put each word and its predicted tag in a new line for the output file 
                words_with_pos_tags=''
                for i in range(len(sentence)): #for all words in sentence
                    words_with_pos_tags += sentence[i] + ' ' + tags[i][1] + '\n' #tags[i][1] gets the pos tag from the tuple of (word, postag)
                
                #all words in sentence over. Before we move to next sentence, add "\n" after the last word prediction
                words_with_pos_tags += '\n'
                
                list_of_all_predictions.append(words_with_pos_tags)

            
            
            # Write final output to output file
            
            with open(output_file_path, 'w') as output_file:
                   output_file.writelines(list_of_all_predictions)
            
            # Calculate accuracy

            accuracy_on_test = calculateAccuracy(predicted_tags, testing_sentences)
            return accuracy_on_test
  
    except Exception as e:
            print(e)













'''

Function to train and tune the tagger by building models with all combinations of hyper-parameters
on train data, evaluating them on cross validation data and then selecting which hyper-parameters are 
perfoming the best (giving max accuracy)

'''



def train():

    try:

        
        # Read input from "train.txt" data file, call preprocess function we have defined above
        all_training_sentences= pre_process_data(input_train_data_path)
        
        # Split data into training and validation set in the ratio 80:20
        train_set,cross_val_set =train_test_split(all_training_sentences,train_size=0.80,test_size=0.20,random_state = 101)
        




        if(tagger_type=="hmm"):

            ''''

            Build Model with default parameters

            '''

            # Train HMM Tagger on this data
            hmm_tagger = train_hmm_tagger(train_set)
            
            # Evaluate HMM Tagger on cross_val_set
            accuracy_on_cv =evaluate_model(hmm_tagger, cross_val_set)








            ''''
            
            #Tuning the HMM Model:



            We are trying the following estimators:

            MLEProbDist (Lidstone with gamma = 0.1) - default
            LidstoneProbDist (gamma==0.001)
            LaplaceProbDist (Lidstone with gamma==1)
            ELEProbDist (= Lidstone with gamma==0.5)
            WittenBellProbDist

            '''



            accuracies_per_estimator={}

            #add the accuracy of default estimator into this dictionary before we try other estimators
            accuracies_per_estimator[None]=accuracy_on_cv

            estimators = [LidstoneProbDist, MLEProbDist, LaplaceProbDist, ELEProbDist, WittenBellProbDist]
            for e in estimators:
                #train hmm with given estimator e as parameter
                hmm_tagger_estimator = train_hmm_tagger(train_set, estimator_hmm=e)
                #evaluate performance of this tagger on cross_val
                accuracy_on_cv= evaluate_model(hmm_tagger_estimator, cross_val_set)
                #add accuracy to dictionary in form of key-value
                accuracies_per_estimator[e]=accuracy_on_cv

            print(accuracies_per_estimator)
            
            #find max accuracy estimator from this dictionary
            maxAccuracyEstimator= max(accuracies_per_estimator, key=accuracies_per_estimator.get)
            print("Max Accuracy is for the estimator: ", maxAccuracyEstimator)
            print("Max Accuracy value: ", accuracies_per_estimator[maxAccuracyEstimator])


            return [maxAccuracyEstimator]  


            








        else: #tagger_type = "brill"
            

            # Train Brill Tagger with default parameters

            template_base= [Template(Pos([-1])), Template(Pos([-1]), Word([0]))] # taken from official documentation

            brill_tagger = train_brill_tagger(train_set, template_arg=template_base)

            

            # Evaluate Brill Tagger on cross_val_set
            
            accuracy_on_cv= evaluate_model(brill_tagger, cross_val_set)
            
            

            '''


            # TUNE BRILL TAGGER MODEL


            1. TEMPLATE

                We are trying the following functions that returns rules for the TEMPLATE parameter:

                nltkdemo18()
                nltkdemo18plus()
                fntbl37()
                brill24()

            2. MAX_RULES
                
                We are trying the following values for max_rules parameter:

                10
                50
                75
                100
            
            '''



            accuracies_per_template_and_rule_count=[]

            #add the accuracy of default template into the result list before we try other estimators
            accuracies_per_template_and_rule_count.append( ["template_base", 10, accuracy_on_cv] )

            
            templates=[["nltkdemo18",nltkdemo18()],["nltkdemo18plus",nltkdemo18plus()],["fntbl37", fntbl37()],["brill24",brill24()]]
            max_rules = [10, 50, 75, 100]

            for temp in templates:

                for ruleCount in max_rules:

                    #train hmm with given template temp, given max_rules number as parameters
                    brill_tagger_for_template = train_brill_tagger(train_set, template_arg= temp[1], max_rules_arg=ruleCount)

                    #evaluate performance of this tagger on cross_val
                    accuracy_on_cv= evaluate_model(brill_tagger_for_template, cross_val_set)

                    #add accuracy to list of results with 3 values: name_of_template, max_rules, accuracy
                    accuracies_per_template_and_rule_count.append( [ temp[0], ruleCount, accuracy_on_cv ] )

                
            print(accuracies_per_template_and_rule_count)

            #find max accuracy template from this list
            maxAccuracyValue=accuracies_per_template_and_rule_count[0][2]
            maxAccuracyTemplateName=''
            maxAccuracyRuleCount=10

            for result in accuracies_per_template_and_rule_count:
                if(result[2]) > maxAccuracyValue:
                    maxAccuracyValue = result[2]
                    maxAccuracyTemplateName=result[0]
                    maxAccuracyRuleCount= result[1]
                    
            
            print("Max Accuracy is for the template: ", maxAccuracyTemplateName)
            print("Max Accuracy is for the rule count: ", maxAccuracyRuleCount)
            print("Max Accuracy value: ", maxAccuracyValue)
           
            #get the template function corresponding to maxAccuracyTemplateName
            #for i in templates:
            #    if(i[0]==maxAccuracyTemplateName):
            #        maxAccuracyTemplateFunction = i[1]
            #        print(i[0])       

            #print(maxAccuracyTemplateFunction)



            return [maxAccuracyTemplateName, maxAccuracyRuleCount]



        
    except Exception as e:
        print(e)











def main():

    try:

        print("=========================================================")

        # Read input from "train.txt" data file, call preprocess function we have defined above
        all_training_sentences= pre_process_data(input_train_data_path)
        

        
        #we need if-else conditions because train() function we made will return different hyperparameters according to which tagger we are training
        if(tagger_type=='hmm'):
            

            ''''
            
            Note: Uncomment the 2 lines below if you want to run code of HYPER-PARAMETER TUNING 
            
            '''
            
            #This function that will train the models and find the optimal parameters
            #print("===========TRAINING HAS STARTED=============")


            #optimal_parameters = train()
            #optimal_estimator= optimal_parameters[0]




            '''
            From the results of hyper-parameter tuning,
            we found that the following estimator was performing the best. 
            So let's use it for our FINAL Model.
            
            '''

            optimal_estimator = WittenBellProbDist
            
            


            # Now finally train this tuned model (with optimal hyper-parameters) on FULL TRAINING

            final_tagger = train_hmm_tagger(all_training_sentences, estimator_hmm=optimal_estimator)






            

        else: #tagger_type="brill"


            ''''
            
            Note: Uncomment the 5 lines below if you want to run code of HYPER-PARAMETER TUNING 
            
            '''


            #This function that will train the models and find the optimal parameters
            #print("===========TRAINING HAS STARTED=============")


            #optimal_parameters = train()
            #optimal_templateName= optimal_parameters[0]
            #optimal_ruleCount= optimal_parameters[1]

            #dictOfTemplates={"nltkdemo18":nltkdemo18(),"nltkdemo18plus":nltkdemo18plus(),"fntbl37": fntbl37(),"brill24":brill24()}
            #optimalTemplateFunction= dictOfTemplates[optimal_templateName]    #get the template function corresponding to optimal_templateName
            
            
            






            '''
            From the results of hyper-parameter tuning,
            we found that the following value of template function and max_rule was performing the best. 
            So let's use these for our FINAL Model.
            
            '''
            
            optimalTemplateFunction = brill24()
            optimal_ruleCount = 100
            
            
            # Now finally train this tuned model (with optimal hyper-parameters) on FULL TRAINING DATA

            final_tagger = train_brill_tagger(all_training_sentences, template_arg= optimalTemplateFunction, max_rules_arg=optimal_ruleCount )

        
        






        
        print("=================FINAL MODEL==============================")

        #Call Function to find predictions on test data, write predictions to output file and get final accuracy
        accuracy_on_test= get_predictions_on_test_data(final_tagger)

        print("Accuracy on test data for "+tagger_type+" tagger is:",accuracy_on_test)

        
    except Exception as e:
        print(e)






if __name__ == "__main__":
    main()




