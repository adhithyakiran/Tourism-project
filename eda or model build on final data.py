#%%


import os
import pandas as pd
import numpy as np



from sklearn.cluster import KMeans

from sklearn.feature_extraction.text import TfidfVectorizer

#%%

#Merging all csv files into a single dataframe

df = pd.concat(
    map(pd.read_csv, ['a (1).csv', 'a (2).csv','a (3).csv','a (4).csv','a (5).csv','a (6).csv','a (7).csv','a (8).csv','a (9).csv','a (10).csv','a (11).csv','a (12).csv','a (13).csv','a (14).csv','a (15).csv','a (16).csv','a (17).csv','a (18).csv','a (19).csv','a (20).csv','a (21).csv']),ignore_index=True)


#%%

#remove first column with row number

df = df.iloc[: , 1:]


#%%

np.random.seed(777)


#%%
df.rename(columns = {'Review ':'Review'}, inplace = True)

#%%
r = df[['Review']]

#%%

r.head()



#%%
#LDA
# We apply a count vectoriser to represent text as numbers as any algorithms expect


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')

dtm = cv.fit_transform(r['Review'])

dtm

#%%

#We import the LDA from sklearn and define the number of clusters i.e n_components = 7

from sklearn.decomposition import LatentDirichletAllocation
LDA = LatentDirichletAllocation(n_components=7,random_state=42)
LDA.fit(dtm)


#%%




#%% [markdown]

# # Let’s display the top 15 words in each 7 clusters we defined.

for index,topic in enumerate(LDA.components_):
    print(f'THE TOP 15 WORDS FOR TOPIC #{index}')
    print([cv.get_feature_names()[i] for i in topic.argsort()[-15:]])
    print('\n')


#%%
#KMeans

#%%

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')

X = vectorizer.fit_transform(r['Review'])

#%%


number_of_clusters = 7

from sklearn.cluster import KMeans


# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
model = KMeans(n_clusters=number_of_clusters, 
               init='k-means++', 
               max_iter=100, # Maximum number of iterations of the k-means algorithm for a single run.
               n_init=1)  # Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.

model.fit(X)


#%%



order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()


#%%

for i in range(number_of_clusters):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind])



#%% [markdown]


# # sentimental analysis


import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#%%


sid = SentimentIntensityAnalyzer()
r['sid_score'] = r['Review'].apply(lambda review: sid.polarity_scores(review))

## take compund values on another variable (for save the values) and delete it for take the lables
x = []
for item in r['sid_score']:
    x.append(item.pop('compound'))
    
## take label for our reviews   
list_score = []
for item in r['sid_score'] :
    list_score.append(max(item, key=item.get))

## create columns compund score and lable for each reviews
r['compound_score'] = x
r['sentiment'] = list_score

x = []
for item in r.compound_score :
    if item >= 0.05 :
        x.append('pos')
    elif item <= -0.05 :
        x.append('neg')
    elif -0.05 < item < 0.05:
        x.append('neu')

r.sentiment = x


#%%

#applying to the entire data set.


d = df 

import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()
d['sid_score'] = d['Review'].apply(lambda review: sid.polarity_scores(review))

## take compund values on another variable (for save the values) and delete it for take the lables
x = []
for item in d['sid_score']:
    x.append(item.pop('compound'))
    
## take label for our reviews   
list_score = []
for item in d['sid_score'] :
    list_score.append(max(item, key=item.get))

## create columns compund score and lable for each reviews
d['compound_score'] = x
d['sentiment'] = list_score

x = []
for item in r.compound_score :
    if item >= 0.05 :
        x.append('pos')
    elif item <= -0.05 :
        x.append('neg')
    elif -0.05 < item < 0.05:
        x.append('neu')

d.sentiment = x



#%%

#Antelope_Flats


Antelope_Flats = d[d["place"] == 'Antelope_Flats']

#pre-covide
#%%

#date

Antelope_Flats['date'] = pd.to_datetime(Antelope_Flats['date'])
#%%

mask = (Antelope_Flats['date'] > '2015-06-01' ) & (Antelope_Flats['date'] <='2020-03-01')

Antelope_Flats_pre = Antelope_Flats.loc[mask]



mask = (Antelope_Flats['date'] > '2020-03-01' ) & (Antelope_Flats['date'] <='2023-03-01')


Antelope_Flats_post = Antelope_Flats.loc[mask]

#%%

#seperate for positive and negative

Antelope_Flats_pre_pos = Antelope_Flats_pre[Antelope_Flats_pre["sentiment"] == 'pos']

#%%

Antelope_Flats_pre_neg = Antelope_Flats_pre[Antelope_Flats_pre["sentiment"] == 'neg']

#%%

Antelope_Flats_post_pos = Antelope_Flats_post[Antelope_Flats_post["sentiment"] == 'pos']

#%%
Antelope_Flats_post_neg = Antelope_Flats_post[Antelope_Flats_post["sentiment"] == 'neg']


#%%

#work cloud

## Lematisasi
from nltk.stem.wordnet import WordNetLemmatizer
## stopwords
from nltk.corpus import stopwords
## punctuation
import string

import nltk
nltk.download('stopwords')
nltk.download('wordnet')

stop = set(stopwords.words('english'))
punc = set(string.punctuation)
lemma = WordNetLemmatizer()

#%%

#keywords in list

keywords = df['place'].apply(lambda x : x.lower()).unique().tolist()

keywords.append('park')
keywords.append('visit')

keywords.append('Flats')
keywords.append('Flat')

keywords.append('flat')
keywords.append('get')
keywords.append('grand')
keywords.append('could')
keywords.append('else')
keywords.append('got')

#%%


### function clean_text : 
def clean_text(text):
    ## convert the text to lowercase
    text = text.lower()
    ## split text to the list 
    wordList = text.split()
    ## remove punctuation
    wordList = ["".join(x for x in word if (x == "'") | (x not in punc)) for word in wordList]
    ## remove stopwords 
    wordList = [word for word in wordList if word not in stop]
    ## remove keywords 
    wordList = [word for word in wordList if word not in keywords]
    ## Lemmatisation 
    wordList = [lemma.lemmatize(word) for word in wordList]
    return " ".join(wordList)

#%%

Antelope_Flats_pre_pos['Review'] = Antelope_Flats_pre_pos['Review'].astype('str')

#%%




#%%

Antelope_Flats_pre_pos['clean_text'] = Antelope_Flats_pre_pos['Review'].apply(clean_text)



#%%

#wordcloud

from wordcloud import WordCloud 

def word_frequency(text):
    wordList = text.split()
    # generate frequencey of word to dictionary 
    wordFreq = {word : wordList.count(word) for word in wordList}
    return wordFreq


word_frequency('I love you, I love my mom, I love my dad, I love him, you  are my everything')

#%%

def wordcloud_freq(word_freq,title,figure_size = (10,5)):
    wordcloud.generate_from_frequencies(word_freq)
    plt.figure(figsize=figure_size)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.title(title)
    plt.show

#%% [markdown]

# # Precovid
#  
# Precovid possitive review's wordcloud for Antelope_Flats

import matplotlib.pyplot as plt 
import seaborn as sns 
import warnings 
warnings.filterwarnings('ignore')

Antelope_Flats_pre_pos_df = " ".join(Antelope_Flats_pre_pos[Antelope_Flats_pre_pos["sentiment"] == 'pos']['clean_text'][0:1000])

Antelope_Flats_pre_pos_df = word_frequency(Antelope_Flats_pre_pos_df)
wordcloud = WordCloud(width = 5000,
                     height = 2500,
                     colormap ='inferno',
                     background_color='white')
wordcloud_freq(Antelope_Flats_pre_pos_df, 'Most Frequent Words in the Latest 1000 Precovid possitive review wordcloud for Antelope_Flats')    


#%% [markdown]

#Precovid review's wordcloud for Antelope_Flats

Antelope_Flats_pre_neg['Review'] = Antelope_Flats_pre_neg['Review'].astype('str')

#%%




#%%

Antelope_Flats_pre_neg['clean_text'] = Antelope_Flats_pre_neg['Review'].apply(clean_text)


Antelope_Flats_pre_neg_df = " ".join(Antelope_Flats_pre_neg[Antelope_Flats_pre_neg["sentiment"] == 'neg']['clean_text'][0:1000])

Antelope_Flats_pre_neg_df = word_frequency(Antelope_Flats_pre_neg_df)
wordcloud = WordCloud(width = 5000,
                     height = 2500,
                     colormap ='Blues',
                     background_color='black')
wordcloud_freq(Antelope_Flats_pre_neg_df, 'Most Frequent Words in the Latest 1000 Precovid negative review wordcloud for Antelope_Flats') 

#%% [markdown]

# # post covid

# post possitive




Antelope_Flats_post_pos['clean_text'] = Antelope_Flats_post_pos['Review'].apply(clean_text)



Antelope_Flats_post_pos_df = " ".join(Antelope_Flats_post_pos[Antelope_Flats_post_pos["sentiment"] == 'pos']['clean_text'][0:1000])

Antelope_Flats_post_pos_df = word_frequency(Antelope_Flats_post_pos_df)
wordcloud = WordCloud(width = 5000,
                     height = 2500,
                     colormap ='inferno',
                     background_color='white')
wordcloud_freq(Antelope_Flats_post_pos_df, 'Most Frequent Words in the Latest 1000 Postcovid possitive review wordcloud for Antelope_Flats')  
















#%% [markdown]

# post-negative


Antelope_Flats_post_neg['Review'] = Antelope_Flats_post_neg['Review'].astype('str')





Antelope_Flats_post_neg['clean_text'] = Antelope_Flats_post_neg['Review'].apply(clean_text)


Antelope_Flats_post_neg_df = " ".join(Antelope_Flats_post_neg[Antelope_Flats_post_neg["sentiment"] == 'neg']['clean_text'][0:1000])

Antelope_Flats_post_neg_df = word_frequency(Antelope_Flats_post_neg_df)
wordcloud = WordCloud(width = 5000,
                     height = 2500,
                     colormap ='Blues',
                     background_color='black')
wordcloud_freq(Antelope_Flats_post_neg_df, 'Most Frequent Words in the Latest 1000 Postcovid negtive review wordcloud for Antelope_Flats') 