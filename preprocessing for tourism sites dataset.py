# %%

import string
from urllib import response
import pandas as pd
import re 
import requests

# %%


df = pd.read_csv('Yellowstone_National_Park2.csv')
# %%

df.columns =['Date (with category)', 'place ', 'Tagline', 'Review ']


#%%

df = df.dropna()

#%%




# %%
category = []

date = []



j = 0

s = 'not available'

for e in df['Date (with category)']:

    x = e.split("•")
    #print(x)

    date.append(x[0])

    if (len(x) > 1 ):

        category.append(x[1])
        j = j+1

    else : 

        category.append(s)
        j = j+1




    #ulist.append(re.findall('http[a-zA-Z/:.0-9]+', i))


#print(ulist)
# %%

#df['category'] = category

df.insert (0, 'category', category)

df.insert (0, 'date', date)

df.insert (0, 'place', 'Yellowstone_National_Park2')




#df['date'] = date 


#%%

df['place '] = df['place '].str.rstrip('contributions')

#df['Location'] = df['Location'].replace(to_replace ='[nN]ew', value = 'New_', regex = True)

df['place '] = df['place '].str.replace('\d+', '')

#%%

# import re
  
# # Function to clean the names
# def Clean_names(Location_name):
#     # Search for opening bracket in the name followed by
#     # any characters repeated any number of times
#     if re.search('\0-9.*', Location_name):
  
#         # Extract the position of beginning of pattern
#         pos = re.search('\0-9.*', Location_name).start()
  
#         # return the cleaned name
#         return Location_name[:pos]
  
#     else:
#         # if clean up needed return the same name
#         return Location_name
          
# # Updated the city columns
# df['place '] = df['place '].apply(Clean_names)

#%%

df = df.drop('Date (with category)', 1)

#%%

#renaming

df.columns = df.columns.str.replace('place	', 'place')

df.columns = df.columns.str.replace('Review	', 'Review')

#%%

df.to_csv('p/Yellowstone_National_Park2.csv')


#%%




# %%
