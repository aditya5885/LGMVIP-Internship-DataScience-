#!/usr/bin/env python
# coding: utf-8

# In[122]:


#imports TfidfVectorizer,cosine similarity 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[123]:


df = pd.read_csv("Downloads\music.csv")     #loads dataset
df.head()                                   #displays top 5 rows


# In[124]:


df.tail()                    #displays last 5 rows


# In[125]:


df.shape                    #displays total rows,columns


# In[126]:


subset_data = df.head(500)  #takes rows upto 500 in dataset for memory optimization


# In[127]:


#text preprocessing technique that transforms a collection of text documents into a numerical matrix 
#TF" means "Term Frequency," which measures the frequency of a term (word)
#"IDF" means "Inverse Document Frequency," which measures how unique a term

tfidf_vectorizer = TfidfVectorizer(stop_words='english') #tells the vectorizer to ignore common English stop words like "the," "is," "and," etc., as these words has no significant.
subset_data['text'] = subset_data['text'].fillna('')     #handling missing values

# Apply TF-IDF vectorization to the 'text' column
tfidf_matrix = tfidf_vectorizer.fit_transform(subset_data['text'])



# In[153]:


#cosine similarity between each pair of songs
#Cosine similarity is commonly used in recommendation systems,Angle Measure
#useful for content-based recommendation systems, where songs are represented as vectors of content features,Efficiency
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
# Print a subset of the cosine similarity matrix
print("Subset of the cosine similarity matrix:")
print(cosine_sim[:5, :5])  # Print the first 5x5 subset of the matrix, first 5 rows and first 5 columns of the matrix




# In[154]:


def get_recommendations(song_title, cosine_similarities=cosine_sim):
    idx = subset_data[subset_data['song'] == song_title].index[0]     # Get the index of the input song
    sim_scores = list(enumerate(cosine_similarities[idx]))            # Get cosine similarities for the input song
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True) # Sort songs by similarity
    sim_scores = sim_scores[1:20]                                     # Get top 20 similar songs (excluding the input song)
    song_indices = [i[0] for i in sim_scores]                         # Get indices of similar songs
    return subset_data['song'].iloc[song_indices]                     # Return recommended song titles



# In[155]:


print(subset_data['song'][:50])    #displays 50 songs titles accordingly


# In[156]:


print("Song titles in the subset data:")
print(subset_data['song'])


# In[157]:


# Test the recommendation function
input_song = "Sleigh Ride"  # Choose a song title from the subset
recommended_songs = get_recommendations(input_song)

print(f"Recommended songs for '{input_song}':")
print(recommended_songs[:20])  # Print the first 20 recommended songs


# In[164]:


input_song = "Dancing Queen"   #Choose a song title from the subset
recommended_songs = get_recommendations(input_song)

print(f"Recommended songs for '{input_song}':")

print(recommended_songs[:10])  # Print the first 10 recommended songs


# In[ ]:




