
"""
Results for the test file suplied
 So far scored 2000 sentences with 29442 words.
                   Words correct:     
   0. Ground truth:      100.00%    
         1. Simple:       93.62%       
            2. HMM:       94.99%      
----

Calculations of initial state distribution, Transition and Emission probabilities:

    Here , Hidden States are Part of  speech labels, Obseerved variables are Words
    Assumptions : When an unknown word or a transistion sequence or unknow emission is obesevred we substitute 
    with a log of small probability i.e math.log(1.0/1000000000)
    
    The probabilities are namely P(S1), P(Si+1|Si),P(Si+2|Si+1,Si)
    and P(Wi|Si)
    Initial State Distribution:
    
       
  
    For P(S1)(namely the initial state distribution) that is the prior of "part of speech" at position 1,
    we have collected all the POS of the first position in each sentence and put it as a key in a dictionary(being startingProbs{} ) 
    of logarithm of probabilities. 
    Hence we have a total of 12 POS as keys and their corresponding logarithm of 
    probabilities of them appearing in the first position. 
    
    For priors to be considered in positions other than first, I have collected all the POS appearing in the 
    positions other than the first and found out the probabilities. This froms my dictionary allSimplePOSProbs{}.
    This was done to make a distinction in distributions for POS according to their positions so as to get better accuracy.
    
    Transition probabilities:
        
        P(Si+1|Si): (Required for HMM)
        Since P(Si+1|Si) = P(Si+1 , Si)/P(Si), We have found the probability of two POS tags appearing in sequence,
        that is P(Si+1 , Si), for example [noun-adj], P(noun,adj) is the probability of noun-adj pair appearing and 
        is stored as the transition probabilities(in the form of log(P)), transitionProbs{} dictionary.
        And to calculate -->> log(P(Si+1|Si)) = log(P(Si+1 , Si))-log(P(Si)),
        For example, log(P(noun/adj)) = log(P(adj , noun))-log(P(adj)) = transitionProbs["adj_noun"]-allSimplePOSProbs["adj"]
        
        P(Si+2|Si+1,Si): (Requiured for Complex Model)
        Since P(Si+2|Si+1,Si) = P(Si+2,Si+1,Si)/P(Si+1,Si), We have found the probability of three POS tags appearing in sequence,
        that is P(Si+2,Si+1,Si). for example [noun-adj-verb], P(noun,adj,verb) is the probability of noun-adj-verb sequence 
        appearing and is stored as the transition probabilities(in the form of log(P)), transition3Probs{} dictionary.
        And to calculate -->> log(P(Si+2|Si+1,Si)) = log(P(Si+2,Si+1,Si))-log(P(Si+1,Si)),
        As an example, --> log(P(noun/adj,verb)) =
        log(P(noun,adj,verb)) - log(P(adj,verb)) = transition3Probs["noun_adj_verb"]-transition3Probs["adj_verb"]
        
    Emission probabilities:
        
        P(Wi/Si):
            P(Word/Part Of Speech) = probability of word given a part of speech. 
            For each POS we collected all the words that were tagged with this particular POS, and were stored in a 
            dictionary of dictionary whose outer key is POS and inner key is the word tagged with this particular POS.
            For instance, P(found/VERB) = emissionProbs["VERB"]["found"] stored in the form of log(P)


Simple Naive Bayes Approach:
    
    In this we make an naive assumption that each POS at a give position is independent of any other positions.
    That is, P(S1,S2,S3,S4,....,Sn/W1,W2,W3,...,Wn) = P(W1/S1)*P(S1)*P(W2/S2)*P(S2)....*P(Wn/Sn)*P(Sn),
    so at every ith position, we find out the "si" leading to the highest probabilty of P(Si = si/W).
    and output the sequence of the labels.
    
    
HMM Model By Viterbi(Fig 1(a)):
    
    Since each state is dependent on the previous state we would need to calculate the sequence "s1,s2,s3,s4,....,sn" 
    such that the probability P(Si = si|W) leads to maximum.
    That is, P(S1,S2,S3,S4,....,Sn/W1,W2,W3,...,Wn) = P(W1/S1)*P(S1)*P(W2/S2)*P(S2/S1)....*P(Wn/Sn)*P(Sn/Sn-1),
    so at every ith position, we find out the "si" leading to the highest probabilty of P(Si = si/W).
    and output the sequence of the labels.
       
    For every state at position t, we find the max of previous states' probability times the transition probability from the 
    precious state to current state and multiply with emission probability of the current state.
    We store this probability in a dictionary dictOfPostions[position][POS] in the form of log(P), now after 
    we calculate the probability at each position for each state, to backtrack we append in the form of string 
    

     maxPaths[position][POS] = maxPaths[position-1][prevStateLeadingToMax] + "=>" + POS, 
     where POS is the current part of speech.(POS at position "t"), such that we dont have to actually traverse or backtrack
     through the entire trace, rather at the last position find the POS that had the maximum probability and pull the string
     that  maxPaths[LastPosition][maxPOS] had. Hence saving the requirement of return loop.
                
Posterior Calculations:
    Say for a sentence of 4 words S1 = s1, S2 = s2,S3 = s3, S4 = s4 , w1,w2,w3,w4
    Simple :
        P(S1,S2,S3,S4|W1,W2,W3,W4) (proprtional to) P(W1 = w1|S1 =s1)*P(S1 =s1)*P(W2=w2|S2=s2)*P(S2=s2)*P(W3=w3|S3=s3)*P(S3=s3)*P(W4=w4|S4=s4)*P(S4=s4)/P(W1,W2,W3,W4) (neglected as as suggested by prof. Crandall)
        
        in logs , I add the Emission Probabilities and Priors.
    
    HMM Model:(denomminator neglected)
         P(S1,S2,S3,...Sn/W1,W2,W3,Wn) (proprtional to) P(w1|s1)*P(s1)*P(w2|S2)*P(s2|s1)....*P(wn|sn)*P(sn|sn-1)
"""
####
# Put your report here!!
####
import timeit
import time
import random
import math
import itertools as itr
import collections as col
import numpy as np

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#


class Solver:
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    
    
    def __init__(self):
        self.startingProbs = {}
        self.emissionProbs = {}
        self.transitionProbs = {}
        self.transition3Probs = {}
        self.allSimplePOSProbs = {}
        self.SMALL_PROB = math.log(1.0/1000000000)
        self.All_POS = ('det','noun','adj','verb','adp','adv','conj','prt','pron','num')
        self.trained = False
    
    def posterior(self, model, sentence, label):
        ##sentence as tuple of words 
        ##lables as tuple of POS
        if model == "Simple":
             ##P(W1 = w1|S1 =s1)*P(S1 =s1)*P(W2=w2|S2=s2)*P(S2=s2)*P(W3=w3|S3=s3)*P(S3=s3)*P(W4=w4|S4=s4)*P(S4=s4)
            logProb = 0
            for position, (word, POS) in enumerate(zip(sentence,label)):
            
                if position == 0:   
                    logProb += (self.emissionProbs.get(POS,{}).get(word,self.SMALL_PROB) + self.startingProbs.get(POS,self.SMALL_PROB))
                else :
                    logProb += (self.emissionProbs.get(POS,{}).get(word,self.SMALL_PROB) + self.allSimplePOSProbs.get(POS,self.SMALL_PROB))
            
            return logProb
			
        elif model == "HMM":
            
            ###P(w1|s1)*P(s1)*P(w2|S2)*P(s2|s1)....*P(wn|sn)*P(sn|sn-1)
            logProb = 0
            for position, (word , prevPOS, POS) in enumerate(zip(sentence,[label[0]]+list(label),label)): ##(e,e,f,g,h),(e,f,g,h)
            
                if position == 0:   
                    logProb += self.emissionProbs.get(POS,{}).get(word,self.SMALL_PROB) + self.startingProbs.get(POS,self.SMALL_PROB)
                
                    
                else :
                    logProb += self.emissionProbs.get(POS,{}).get(word,self.SMALL_PROB) + self.transitionProbs.get(prevPOS+"_"+POS,self.SMALL_PROB) \
                    -self.allSimplePOSProbs.get(prevPOS,self.SMALL_PROB)
            return logProb
        
            
                
            

        else:
            print("Unknown algo!")

    # Do the training!
    #
    
    def load_data(self, filename):
        
        ##returns list of sentences as each sentence is ( (words),(tags))
        
        AllSentencesAsWandT= [] ## [((words),(corresponding tags)), (sentence 2)]
        file = open(filename, 'r');
        for line in file:
            data = tuple([w.lower() for w in line.split()])
            words = data[0::2]
            tags = data[1::2]
            sentenceAsWordsAndTags = (words,tags)
            AllSentencesAsWandT += [sentenceAsWordsAndTags] #isolating sentence as well
        file.close()
        return AllSentencesAsWandT
    
    
    def train(self, data):
        ##load the Transition, Emission probabilities and Starting Probabilities
        ## data = list of sentences and each sentence is ( (w1,w2),(s1,s2))
        
        ###################P(S1)##########################################
        totalSentences = len(data)
        
        AllS1List = [sentAsWordsTags[1][0] for sentAsWordsTags in data]
        
        Total_S1 = len(AllS1List)
        ProbOfPOS1AsDict = dict(col.Counter(AllS1List)) 
        
        ProbOfPOS1AsDict = {key: math.log(ProbOfPOS1AsDict[key]/Total_S1) for key in ProbOfPOS1AsDict}
        
       ###############(P(S))############################ 
        AllButS1List = list(itr.chain.from_iterable([sentAsWordsTags[1][1:] for sentAsWordsTags in data]))
        Total_SbutS1 = len(AllButS1List)
        ProbOfPOS_AllAsDict = dict(col.Counter(AllButS1List)) 
        ProbOfPOS_AllAsDict = {key: math.log(ProbOfPOS_AllAsDict[key]/Total_SbutS1) for key in ProbOfPOS_AllAsDict} 
        
        
        ###################### transition P(S2/S1) 
        ### and Emission probability : P(Word/PartOfSpeech)##############################
        

        seqTags = []
        ProbWordGivenPOSAsDict = col.defaultdict(list)
        
        for (words,tags) in data:
         ##can be combined in list comprehension but would lose readability
         

             
             
             seqTags += [ tag + "_" + nextTag for tag,nextTag in zip(tags,tags[1:])]
             
             
         
             ##loop for wordGivenTags
             for word,tag in zip(words,tags):
                ProbWordGivenPOSAsDict[tag].append(word)
        
        
        ### for transition probs p(s2/s1)
        TotalNumSequences = len(seqTags)
        
        ProbOfTagSeqAsDict = dict(col.Counter(seqTags))
        ProbOfTagSeqAsDict = {key: math.log(ProbOfTagSeqAsDict[key]/TotalNumSequences) for key in ProbOfTagSeqAsDict}
        
        
        ###for emission probs
        
        for tagKey , wordList in ProbWordGivenPOSAsDict.items():
            
            #transform Wordlist to dict of words and probabilities
            totalWordsInTheTag = len(wordList)
            wordDict = dict(col.Counter(wordList))
            
            ##log of probability
            wordDict = {word: math.log(frequency/totalWordsInTheTag)  for word,frequency in wordDict.items()}
            
            ProbWordGivenPOSAsDict[tagKey] = wordDict
        
        
         ### for transition probs p(s3/s2,s1)
        
        
        
        
        
        self.startingProbs = ProbOfPOS1AsDict
        self.emissionProbs = ProbWordGivenPOSAsDict
        self.transitionProbs = ProbOfTagSeqAsDict
        self.allSimplePOSProbs = ProbOfPOS_AllAsDict
        self.trained = True
        
        
        pass
     
    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #naive bayes
    def simplified(self, sentence):
        maxPOSList = []
        
        for position,word in enumerate(sentence) :
            maxPOS = ''
            maxProb = -float("inf")
            for POS in self.All_POS :
                if position == 0 :
                    
                    currentProb = self.emissionProbs.get(POS,{}).get(word,self.SMALL_PROB) + self.startingProbs.get(POS,self.SMALL_PROB)
                
                else:
                    
                    currentProb = self.emissionProbs.get(POS,{}).get(word,self.SMALL_PROB) + self.allSimplePOSProbs.get(POS,self.SMALL_PROB)
                
                if currentProb > maxProb:
                    maxProb = currentProb
                    maxPOS = POS
            maxPOSList.append(maxPOS)
        
                
        return maxPOSList
		
    def hmm_viterbi(self, sentence):
                
       ##checck if trained 
       lastPosition = len(sentence)-1
       ##sentence is passed as tuple of words
       maxPaths = col.defaultdict(dict)
       dictOfPostions = col.defaultdict(dict) ##postionons and POS probabilities
       for position,word in enumerate(sentence) :
           #for severy postion
           for POS in self.All_POS :
            #for eveery state at a postion
               if position == 0 :
                        
                   dictOfPostions[position][POS] = self.startingProbs.get(POS,self.SMALL_PROB) \
                   + self.emissionProbs.get(POS,{}).get(word,self.SMALL_PROB)
                       
               
                   maxPaths[position][POS] = POS
                               
               else:
                   #max((value, prevPOS))
               
                   (logProbability,prevStateLeadingToMax)  = \
                   max(( (dictOfPostions[position-1][prevPOS]+ self.transitionProbs.get(prevPOS+"_"+POS,self.SMALL_PROB)-self.allSimplePOSProbs.get(prevPOS,self.SMALL_PROB), prevPOS)\
                        for prevPOS in self.All_POS))
    
                
                   dictOfPostions[position][POS] = logProbability + self.emissionProbs.get(POS,{}).get(word,self.SMALL_PROB)
                       
                   
                   maxPaths[position][POS] = maxPaths[position-1][prevStateLeadingToMax] + "=>" + POS
       maxLastProb,lastMaxPOS = max(((probability,lastPOS) for lastPOS,probability in dictOfPostions[lastPosition].items()))
     
       return maxPaths[lastPosition][lastMaxPOS].split("=>")
#                   dictOfPostions[position][POS] = 
                   
               
    

    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")