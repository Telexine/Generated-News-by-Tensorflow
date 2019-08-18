
# Generated News by Tensorflow
------------
Dataset From The publications include the New York Times, Breitbart, CNN, Business Insider, the Atlantic, Fox News, Talking Points Memo, Buzzfeed News, National Review, New York Post, the Guardian, NPR, Reuters, Vox, and the Washington Post. https://www.kaggle.com/snapcrack/all-the-news


# Requirements
------------
- Python 3.6+
- nltk with all packages
- (optional) tensorflow-gpu==2.0.0-beta1
- (optional) cudnn and cuda 10.1 

## Import Library


```python
try:
  # !pip install tensorflow-gpu==2.0.0-beta1
  pass
except Exception:
  pass
import pandas as pd
import tensorflow as tf
import re
import string
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import words
from nltk.corpus import stopwords
porter = PorterStemmer()
import nltk
# uncomment line below for the first time to download the nltk packages
#nltk.download()
import numpy as np
import os
import glob
import time
tf.__version__

```




    '2.0.0-beta1'




```python
#Disable GPU for non nvidia GPU-- uncomment line below
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

```

    GPU found
    

## Data Preparation Function


```python
tk = RegexpTokenizer(r"[a-zA-Z]+")
stop_words = set(stopwords.words('english')) 
flatten = lambda l: [item for sublist in l for item in sublist]

def remove_punc(txt):
    text = "".join([t for t in txt if t not in string.punctuation])
    return text
# Split sentense to list and remove non text
def tokenize(df):
    return tk.tokenize(text=df.lower())
#Lemmatizing to reduce inflectional forms 
lemmatizer = WordNetLemmatizer()
def word_lemmatizer(txt):
    #change to nounce
    lemtxt = [ lemmatizer.lemmatize(word,'n')  for word in txt  ]
    return lemtxt
# Stem Words

def word_stem(txt):
    stemmed = [porter.stem(word) for word in txt]
    return stemmed
def stop_word(txt):
    filtered_sentence = [] 
    for w in txt: 
        if w not in stop_words: 
            filtered_sentence.append(w) 
    return filtered_sentence 
    
```

## Load dataset to pandas


```python
# Depend on GPU ram (word dict size) But I recommended to use 1 file first

df =[]
ar= pd.read_csv('Datasets/articles1.csv')
df.append(ar)
#ar = pd.read_csv('Datasets/articles2.csv')
#df.append(ar)
#ar = pd.read_csv('Datasets/articles3.csv')
#df.append(ar)
df = pd.concat(df, ignore_index=True,sort=False)
```

### Peek at Data


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>id</th>
      <th>title</th>
      <th>publication</th>
      <th>author</th>
      <th>date</th>
      <th>year</th>
      <th>month</th>
      <th>url</th>
      <th>content</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>17283</td>
      <td>House Republicans Fret About Winning Their Hea...</td>
      <td>New York Times</td>
      <td>Carl Hulse</td>
      <td>2016-12-31</td>
      <td>2016.0</td>
      <td>12.0</td>
      <td>NaN</td>
      <td>WASHINGTON  —   Congressional Republicans have...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>17284</td>
      <td>Rift Between Officers and Residents as Killing...</td>
      <td>New York Times</td>
      <td>Benjamin Mueller and Al Baker</td>
      <td>2017-06-19</td>
      <td>2017.0</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>After the bullet shells get counted, the blood...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>17285</td>
      <td>Tyrus Wong, ‘Bambi’ Artist Thwarted by Racial ...</td>
      <td>New York Times</td>
      <td>Margalit Fox</td>
      <td>2017-01-06</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>When Walt Disney’s “Bambi” opened in 1942, cri...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>17286</td>
      <td>Among Deaths in 2016, a Heavy Toll in Pop Musi...</td>
      <td>New York Times</td>
      <td>William McDonald</td>
      <td>2017-04-10</td>
      <td>2017.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Death may be the great equalizer, but it isn’t...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>17287</td>
      <td>Kim Jong-un Says North Korea Is Preparing to T...</td>
      <td>New York Times</td>
      <td>Choe Sang-Hun</td>
      <td>2017-01-02</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>SEOUL, South Korea  —   North Korea’s leader, ...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>17288</td>
      <td>Sick With a Cold, Queen Elizabeth Misses New Y...</td>
      <td>New York Times</td>
      <td>Sewell Chan</td>
      <td>2017-01-02</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>LONDON  —   Queen Elizabeth II, who has been b...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>17289</td>
      <td>Taiwan’s President Accuses China of Renewed In...</td>
      <td>New York Times</td>
      <td>Javier C. Hernández</td>
      <td>2017-01-02</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>BEIJING  —   President Tsai   of Taiwan sharpl...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>17290</td>
      <td>After ‘The Biggest Loser,’ Their Bodies Fought...</td>
      <td>New York Times</td>
      <td>Gina Kolata</td>
      <td>2017-02-08</td>
      <td>2017.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>Danny Cahill stood, slightly dazed, in a blizz...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>17291</td>
      <td>First, a Mixtape. Then a Romance. - The New Yo...</td>
      <td>New York Times</td>
      <td>Katherine Rosman</td>
      <td>2016-12-31</td>
      <td>2016.0</td>
      <td>12.0</td>
      <td>NaN</td>
      <td>Just how   is Hillary Kerr, the    founder of ...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>17292</td>
      <td>Calling on Angels While Enduring the Trials of...</td>
      <td>New York Times</td>
      <td>Andy Newman</td>
      <td>2016-12-31</td>
      <td>2016.0</td>
      <td>12.0</td>
      <td>NaN</td>
      <td>Angels are everywhere in the Muñiz family’s ap...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>17293</td>
      <td>Weak Federal Powers Could Limit Trump’s Climat...</td>
      <td>New York Times</td>
      <td>Justin Gillis</td>
      <td>2017-01-03</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>With Donald J. Trump about to take control of ...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>17294</td>
      <td>Can Carbon Capture Technology Prosper Under Tr...</td>
      <td>New York Times</td>
      <td>John Schwartz</td>
      <td>2017-01-05</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>THOMPSONS, Tex.  —   Can one of the most promi...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>17295</td>
      <td>Mar-a-Lago, the Future Winter White House and ...</td>
      <td>New York Times</td>
      <td>Maggie Haberman</td>
      <td>2017-01-02</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>WEST PALM BEACH, Fla.  —   When   Donald J. Tr...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>17296</td>
      <td>How to form healthy habits in your 20s - The N...</td>
      <td>New York Times</td>
      <td>Charles Duhigg</td>
      <td>2017-01-02</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>This article is part of a series aimed at help...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14</td>
      <td>17297</td>
      <td>Turning Your Vacation Photos Into Works of Art...</td>
      <td>New York Times</td>
      <td>Stephanie Rosenbloom</td>
      <td>2017-04-14</td>
      <td>2017.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>It’s the season for family travel and photos  ...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15</td>
      <td>17298</td>
      <td>As Second Avenue Subway Opens, a Train Delay E...</td>
      <td>New York Times</td>
      <td>Emma G. Fitzsimmons</td>
      <td>2017-01-02</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>Finally. The Second Avenue subway opened in Ne...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>16</td>
      <td>17300</td>
      <td>Dylann Roof Himself Rejects Best Defense Again...</td>
      <td>New York Times</td>
      <td>Kevin Sack and Alan Blinder</td>
      <td>2017-01-02</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>pages into the   journal found in Dylann S. ...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>17</td>
      <td>17301</td>
      <td>Modi’s Cash Ban Brings Pain, but Corruption-We...</td>
      <td>New York Times</td>
      <td>Geeta Anand</td>
      <td>2017-01-02</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>MUMBAI, India  —   It was a bold and risky gam...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>18</td>
      <td>17302</td>
      <td>Suicide Bombing in Baghdad Kills at Least 36 -...</td>
      <td>New York Times</td>
      <td>The Associated Press</td>
      <td>2017-01-03</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>BAGHDAD  —   A suicide bomber detonated a pick...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>19</td>
      <td>17303</td>
      <td>Fecal Pollution Taints Water at Melbourne’s Be...</td>
      <td>New York Times</td>
      <td>Brett Cole</td>
      <td>2017-01-03</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>SYDNEY, Australia  —   The annual beach pilgri...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>20</td>
      <td>17305</td>
      <td>N.F.L. Playoffs: Schedule, Matchups and Odds -...</td>
      <td>New York Times</td>
      <td>Benjamin Hoffman</td>
      <td>2017-01-03</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>When the Green Bay Packers lost to the Washing...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>21</td>
      <td>17306</td>
      <td>Mariah Carey’s Manager Blames Producers for Ne...</td>
      <td>New York Times</td>
      <td>Patrick Healy</td>
      <td>2017-01-02</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>Mariah Carey suffered through a performance tr...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>22</td>
      <td>17307</td>
      <td>Damaged by War, Syria’s Cultural Sites Rise An...</td>
      <td>New York Times</td>
      <td>Marlise Simons</td>
      <td>2017-01-01</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>PARIS  —   When the Islamic State was about to...</td>
    </tr>
    <tr>
      <th>23</th>
      <td>23</td>
      <td>17308</td>
      <td>George Michael’s Freedom Video: An Oral Histor...</td>
      <td>New York Times</td>
      <td>Guy Trebay and Jacob Bernstein</td>
      <td>2017-01-01</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>Pop music and fashion never met cuter than in ...</td>
    </tr>
    <tr>
      <th>24</th>
      <td>24</td>
      <td>17309</td>
      <td>With New Congress Poised to Convene, Obama’s P...</td>
      <td>New York Times</td>
      <td>Jennifer Steinhauer</td>
      <td>2017-01-02</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>WASHINGTON  —   The most powerful and ambitiou...</td>
    </tr>
    <tr>
      <th>25</th>
      <td>25</td>
      <td>17311</td>
      <td>Republicans Stonewalled Obama. Now the Ball Is...</td>
      <td>New York Times</td>
      <td>Carl Hulse</td>
      <td>2017-01-03</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>WASHINGTON  —   It’s   or   time for Republica...</td>
    </tr>
    <tr>
      <th>26</th>
      <td>26</td>
      <td>17312</td>
      <td>Istanbul, Donald Trump, Benjamin Netanyahu: Yo...</td>
      <td>New York Times</td>
      <td>Charles McDermid</td>
      <td>2017-01-03</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>Good morning.  Here’s what you need to know: •...</td>
    </tr>
    <tr>
      <th>27</th>
      <td>27</td>
      <td>17313</td>
      <td>Inside Trump Defense Secretary Pick’s Efforts ...</td>
      <td>New York Times</td>
      <td>Sheri Fink and Helene Cooper</td>
      <td>2017-01-05</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>The body of the Iraqi prisoner was found naked...</td>
    </tr>
    <tr>
      <th>28</th>
      <td>28</td>
      <td>17314</td>
      <td>ISIS Claims Responsibility for Istanbul Nightc...</td>
      <td>New York Times</td>
      <td>Tim Arango</td>
      <td>2017-01-03</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>ISTANBUL  —   The Islamic State on Monday issu...</td>
    </tr>
    <tr>
      <th>29</th>
      <td>29</td>
      <td>17317</td>
      <td>The Afghan War and the Evolution of Obama - Th...</td>
      <td>New York Times</td>
      <td>Mark Landler</td>
      <td>2017-01-17</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>WASHINGTON  —   President Obama’s advisers wre...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>49970</th>
      <td>53262</td>
      <td>73440</td>
      <td>Barack Obama’s Enduring Faith in America</td>
      <td>Atlantic</td>
      <td>David A. Graham</td>
      <td>2017-01-10</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>In his final speech to the nation as the 44th ...</td>
    </tr>
    <tr>
      <th>49971</th>
      <td>53263</td>
      <td>73441</td>
      <td>Michael Cohen: ’It Is Fake News Meant to Malig...</td>
      <td>Atlantic</td>
      <td>Rosie Gray</td>
      <td>2017-01-10</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>Donald Trump and his lawyer on Tuesday night d...</td>
    </tr>
    <tr>
      <th>49972</th>
      <td>53264</td>
      <td>73442</td>
      <td>The Atlantic Daily: Change and Confirmation</td>
      <td>Atlantic</td>
      <td>Rosa Inocencio Smith</td>
      <td>2017-01-10</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>This article is part of a feature we a...</td>
    </tr>
    <tr>
      <th>49973</th>
      <td>53265</td>
      <td>73443</td>
      <td>What CNN’s Report on Trump and Russia Does and...</td>
      <td>Atlantic</td>
      <td>David A. Graham</td>
      <td>2017-01-10</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>Updated on January 10 at 6:36 p. m.  Despite a...</td>
    </tr>
    <tr>
      <th>49974</th>
      <td>53266</td>
      <td>73444</td>
      <td>The Atlantic  Politics &amp; Policy Daily: Obama Out</td>
      <td>Atlantic</td>
      <td>Candice Norwood</td>
      <td>2017-01-10</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>This article is part of a feature we a...</td>
    </tr>
    <tr>
      <th>49975</th>
      <td>53267</td>
      <td>73445</td>
      <td>What the World Might Look Like in 5 Years, Acc...</td>
      <td>Atlantic</td>
      <td>Uri Friedman</td>
      <td>2017-01-10</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>Every four years, a group of U. S. intelligenc...</td>
    </tr>
    <tr>
      <th>49976</th>
      <td>53268</td>
      <td>73446</td>
      <td>The U.S. Supreme Court Puts North Carolina’s 2...</td>
      <td>Atlantic</td>
      <td>David A. Graham</td>
      <td>2017-01-10</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>DURHAM, N. C. —  The Supreme Court has a messa...</td>
    </tr>
    <tr>
      <th>49977</th>
      <td>53269</td>
      <td>73447</td>
      <td>Trump Meets With Vaccine Skeptic, Discusses ’C...</td>
      <td>Atlantic</td>
      <td>Julie Beck</td>
      <td>2017-01-10</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>Updated on January 10, 8:55 p. m. On Tuesday, ...</td>
    </tr>
    <tr>
      <th>49978</th>
      <td>53270</td>
      <td>73448</td>
      <td>Can the Flaws in Credit Scoring Be Fixed?</td>
      <td>Atlantic</td>
      <td>Gillian B. White</td>
      <td>2017-01-10</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>That credit scoring and reporting is an opaque...</td>
    </tr>
    <tr>
      <th>49979</th>
      <td>53271</td>
      <td>73449</td>
      <td>Trump’s Cyber-Appeasement Policy Might Encoura...</td>
      <td>Atlantic</td>
      <td>Kaveh Waddell</td>
      <td>2017-01-10</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>Since well before he was elected president, Do...</td>
    </tr>
    <tr>
      <th>49980</th>
      <td>53272</td>
      <td>73450</td>
      <td>Taboo: A Grim, Gruesome Costume Drama Starring...</td>
      <td>Atlantic</td>
      <td>Sophie Gilbert</td>
      <td>2017-01-10</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>Nobody excels at playing ferocious psychopaths...</td>
    </tr>
    <tr>
      <th>49981</th>
      <td>53273</td>
      <td>73451</td>
      <td>Clare Hollingworth: The Reporter Who Broke the...</td>
      <td>Atlantic</td>
      <td>David A. Graham</td>
      <td>2017-01-10</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>Any big journalistic scoop requires a combinat...</td>
    </tr>
    <tr>
      <th>49982</th>
      <td>53274</td>
      <td>73452</td>
      <td>The Gaps in New York’s Free-College Plan</td>
      <td>Atlantic</td>
      <td>James S. Murphy</td>
      <td>2017-01-10</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>New York Governor Andrew Cuomo recently announ...</td>
    </tr>
    <tr>
      <th>49983</th>
      <td>53275</td>
      <td>73453</td>
      <td>‘We Have a Problem’: John Kerry on Making Poli...</td>
      <td>Atlantic</td>
      <td>Uri Friedman</td>
      <td>2017-01-10</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>In one of his last public appearances as U. S....</td>
    </tr>
    <tr>
      <th>49984</th>
      <td>53276</td>
      <td>73454</td>
      <td>The Enduring Mystery of Pain Measurement</td>
      <td>Atlantic</td>
      <td>John Walsh</td>
      <td>2017-01-10</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>One night in May, my wife sat up in bed and sa...</td>
    </tr>
    <tr>
      <th>49985</th>
      <td>53277</td>
      <td>73455</td>
      <td>What Conan O’Brien Means to Late Night’s Future</td>
      <td>Atlantic</td>
      <td>David Sims</td>
      <td>2017-01-10</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>Conan O’Brien was once the upstart of the   co...</td>
    </tr>
    <tr>
      <th>49986</th>
      <td>53278</td>
      <td>73456</td>
      <td>The Absurdity of Attacking Celebrities to Defe...</td>
      <td>Atlantic</td>
      <td>Conor Friedersdorf</td>
      <td>2017-01-10</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>Fifty years ago, California Republicans electe...</td>
    </tr>
    <tr>
      <th>49987</th>
      <td>53279</td>
      <td>73457</td>
      <td>Drive-Through Redwoods Are Monuments to Violen...</td>
      <td>Atlantic</td>
      <td>Sarah Zhang</td>
      <td>2017-01-10</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>This weekend, amidst a torrent of rain, one of...</td>
    </tr>
    <tr>
      <th>49988</th>
      <td>53280</td>
      <td>73458</td>
      <td>How Superstar Economics Is Killing the NFL’s R...</td>
      <td>Atlantic</td>
      <td>Derek Thompson</td>
      <td>2017-01-10</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>For years the National Football League has bee...</td>
    </tr>
    <tr>
      <th>49989</th>
      <td>53281</td>
      <td>73459</td>
      <td>The Atlantic Daily: Passing the Presidential Mic</td>
      <td>Atlantic</td>
      <td>Rosa Inocencio Smith</td>
      <td>2017-01-11</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>This article is part of a feature we a...</td>
    </tr>
    <tr>
      <th>49990</th>
      <td>53282</td>
      <td>73460</td>
      <td>How Blackmail Works in Russia</td>
      <td>Atlantic</td>
      <td>Julia Ioffe</td>
      <td>2017-01-11</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>In January 1999, Prosecutor General Yury Skura...</td>
    </tr>
    <tr>
      <th>49991</th>
      <td>53283</td>
      <td>73461</td>
      <td>The Atlantic  Politics &amp; Policy Daily: Back-to...</td>
      <td>Atlantic</td>
      <td>Candice Norwood</td>
      <td>2017-01-11</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>This article is part of a feature we a...</td>
    </tr>
    <tr>
      <th>49992</th>
      <td>53284</td>
      <td>73462</td>
      <td>Obama Built an ‘Infrastructure’ for Civil-Libe...</td>
      <td>Atlantic</td>
      <td>Emma Green</td>
      <td>2017-01-11</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>President Obama’s farewell speech was an exerc...</td>
    </tr>
    <tr>
      <th>49993</th>
      <td>53285</td>
      <td>73463</td>
      <td>Why Trump’s Conflict-of-Interest Plan Won’t Pr...</td>
      <td>Atlantic</td>
      <td>Clare Foran</td>
      <td>2017-01-11</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>Updated on January 11 at 5:56 p. m. ET,   Dona...</td>
    </tr>
    <tr>
      <th>49994</th>
      <td>53286</td>
      <td>73464</td>
      <td>The Irrationally Divided Critics of Donald Trump</td>
      <td>Atlantic</td>
      <td>Conor Friedersdorf</td>
      <td>2017-01-11</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>A large cohort of Americans have reservations ...</td>
    </tr>
    <tr>
      <th>49995</th>
      <td>53287</td>
      <td>73465</td>
      <td>Rex Tillerson Says Climate Change Is Real, but …</td>
      <td>Atlantic</td>
      <td>Robinson Meyer</td>
      <td>2017-01-11</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>As chairman and CEO of ExxonMobil, Rex Tillers...</td>
    </tr>
    <tr>
      <th>49996</th>
      <td>53288</td>
      <td>73466</td>
      <td>The Biggest Intelligence Questions Raised by t...</td>
      <td>Atlantic</td>
      <td>Amy Zegart</td>
      <td>2017-01-11</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>I’ve spent nearly 20 years looking at intellig...</td>
    </tr>
    <tr>
      <th>49997</th>
      <td>53289</td>
      <td>73467</td>
      <td>Trump Announces Plan That Does Little to Resol...</td>
      <td>Atlantic</td>
      <td>Jeremy Venook</td>
      <td>2017-01-11</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>Donald Trump will not be taking necessary st...</td>
    </tr>
    <tr>
      <th>49998</th>
      <td>53290</td>
      <td>73468</td>
      <td>Dozens of For-Profit Colleges Could Soon Close</td>
      <td>Atlantic</td>
      <td>Emily DeRuy</td>
      <td>2017-01-11</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>Dozens of   colleges could be forced to close ...</td>
    </tr>
    <tr>
      <th>49999</th>
      <td>53291</td>
      <td>73469</td>
      <td>The Milky Way’s Stolen Stars</td>
      <td>Atlantic</td>
      <td>Marina Koren</td>
      <td>2017-01-11</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>The force of gravity can be described using a ...</td>
    </tr>
  </tbody>
</table>
<p>50000 rows × 10 columns</p>
</div>



## Applied Cleansing Vocab Function


```python
#df2 =  list(filter(regex.search,df['content'].head().str.cat(sep=' ').split(" "))) 200
#df2 =  df['content'].apply(lambda x: remove_punc(x)).head(1).apply(lambda x:tokenize(x)).apply(lambda x: word_lemmatizer(x)  ) #.apply(lambda x: word_stem(x))

#Number of news to load (depend on GPU ram size)
articles = 30
df2 = df['content'].apply(lambda x: remove_punc(x)).head(articles).apply(tokenize).apply(word_lemmatizer).apply(stop_word) #.apply( word_stem)
vocab = flatten(df2.values.tolist())
del df
```


```python
vocab
```




    ['washington',
     'congressional',
     'republican',
     'new',
     'fear',
     'come',
     'health',
     'care',
     'lawsuit',
     'obama',
     'administration',
     'might',
     'win',
     'incoming',
     'trump',
     'administration',
     'could',
     'choose',
     'longer',
     'defend',
     'executive',
     'branch',
     'suit',
     'challenge',
     'administration',
     'authority',
     'spend',
     'billion',
     'dollar',
     'health',
     'insurance',
     'subsidy',
     'american',
     'handing',
     'house',
     'republican',
     'big',
     'victory',
     'issue',
     'sudden',
     'loss',
     'disputed',
     'subsidy',
     'could',
     'conceivably',
     'cause',
     'health',
     'care',
     'program',
     'implode',
     'leaving',
     'million',
     'people',
     'without',
     'access',
     'health',
     'insurance',
     'republican',
     'prepared',
     'replacement',
     'could',
     'lead',
     'chaos',
     'insurance',
     'market',
     'spur',
     'political',
     'backlash',
     'republican',
     'gain',
     'full',
     'control',
     'government',
     'stave',
     'outcome',
     'republican',
     'could',
     'find',
     'awkward',
     'position',
     'appropriating',
     'huge',
     'sum',
     'temporarily',
     'prop',
     'obama',
     'health',
     'care',
     'law',
     'angering',
     'conservative',
     'voter',
     'demanding',
     'end',
     'law',
     'year',
     'another',
     'twist',
     'donald',
     'j',
     'trump',
     'administration',
     'worried',
     'preserving',
     'executive',
     'branch',
     'prerogative',
     'could',
     'choose',
     'fight',
     'republican',
     'ally',
     'house',
     'central',
     'question',
     'dispute',
     'eager',
     'avoid',
     'ugly',
     'political',
     'pileup',
     'republican',
     'capitol',
     'hill',
     'trump',
     'transition',
     'team',
     'gaming',
     'handle',
     'lawsuit',
     'election',
     'ha',
     'put',
     'limbo',
     'least',
     'late',
     'february',
     'united',
     'state',
     'court',
     'appeal',
     'district',
     'columbia',
     'circuit',
     'yet',
     'ready',
     'divulge',
     'strategy',
     'given',
     'pending',
     'litigation',
     'involves',
     'obama',
     'administration',
     'congress',
     'would',
     'inappropriate',
     'comment',
     'said',
     'phillip',
     'j',
     'blando',
     'spokesman',
     'trump',
     'transition',
     'effort',
     'upon',
     'taking',
     'office',
     'trump',
     'administration',
     'evaluate',
     'case',
     'related',
     'aspect',
     'affordable',
     'care',
     'act',
     'potentially',
     'decision',
     'judge',
     'rosemary',
     'collyer',
     'ruled',
     'house',
     'republican',
     'standing',
     'sue',
     'executive',
     'branch',
     'spending',
     'dispute',
     'obama',
     'administration',
     'distributing',
     'health',
     'insurance',
     'subsidy',
     'violation',
     'constitution',
     'without',
     'approval',
     'congress',
     'justice',
     'department',
     'confident',
     'judge',
     'collyer',
     'decision',
     'would',
     'reversed',
     'quickly',
     'appealed',
     'subsidy',
     'remained',
     'place',
     'appeal',
     'successfully',
     'seeking',
     'temporary',
     'halt',
     'proceeding',
     'mr',
     'trump',
     'house',
     'republican',
     'last',
     'month',
     'told',
     'court',
     'transition',
     'team',
     'currently',
     'discussing',
     'potential',
     'option',
     'resolution',
     'matter',
     'take',
     'effect',
     'inauguration',
     'jan',
     'suspension',
     'case',
     'house',
     'lawyer',
     'said',
     'provide',
     'future',
     'administration',
     'time',
     'consider',
     'whether',
     'continue',
     'prosecuting',
     'otherwise',
     'resolve',
     'appeal',
     'republican',
     'leadership',
     'official',
     'house',
     'acknowledge',
     'possibility',
     'cascading',
     'effect',
     'payment',
     'totaled',
     'estimated',
     'billion',
     'suddenly',
     'stopped',
     'insurer',
     'receive',
     'subsidy',
     'exchange',
     'paying',
     'cost',
     'deductible',
     'eligible',
     'consumer',
     'could',
     'race',
     'drop',
     'coverage',
     'since',
     'would',
     'losing',
     'money',
     'loss',
     'subsidy',
     'could',
     'destabilize',
     'entire',
     'program',
     'cause',
     'lack',
     'confidence',
     'lead',
     'insurer',
     'seek',
     'quick',
     'exit',
     'well',
     'anticipating',
     'trump',
     'administration',
     'might',
     'inclined',
     'mount',
     'vigorous',
     'fight',
     'house',
     'republican',
     'given',
     'dim',
     'view',
     'health',
     'care',
     'law',
     'team',
     'lawyer',
     'month',
     'sought',
     'intervene',
     'case',
     'behalf',
     'two',
     'participant',
     'health',
     'care',
     'program',
     'request',
     'lawyer',
     'predicted',
     'deal',
     'house',
     'republican',
     'new',
     'administration',
     'dismiss',
     'settle',
     'case',
     'produce',
     'devastating',
     'consequence',
     'individual',
     'receive',
     'reduction',
     'well',
     'nation',
     'health',
     'insurance',
     'health',
     'care',
     'system',
     'generally',
     'matter',
     'happens',
     'house',
     'republican',
     'say',
     'want',
     'prevail',
     'two',
     'overarching',
     'concept',
     'congressional',
     'power',
     'purse',
     'right',
     'congress',
     'sue',
     'executive',
     'branch',
     'violates',
     'constitution',
     'regarding',
     'spending',
     'power',
     'house',
     'republican',
     'contend',
     'congress',
     'never',
     'appropriated',
     'money',
     'subsidy',
     'required',
     'constitution',
     'suit',
     'wa',
     'initially',
     'championed',
     'john',
     'boehner',
     'house',
     'speaker',
     'time',
     'later',
     'house',
     'committee',
     'report',
     'republican',
     'asserted',
     'administration',
     'desperate',
     'funding',
     'required',
     'treasury',
     'department',
     'provide',
     'despite',
     'widespread',
     'internal',
     'skepticism',
     'spending',
     'wa',
     'proper',
     'white',
     'house',
     'said',
     'spending',
     'wa',
     'permanent',
     'part',
     'law',
     'passed',
     'annual',
     'appropriation',
     'wa',
     'required',
     'even',
     'though',
     'administration',
     'initially',
     'sought',
     'one',
     'important',
     'house',
     'republican',
     'judge',
     'collyer',
     'found',
     'congress',
     'standing',
     'sue',
     'white',
     'house',
     'issue',
     'ruling',
     'many',
     'legal',
     'expert',
     'said',
     'wa',
     'flawed',
     'want',
     'precedent',
     'set',
     'restore',
     'congressional',
     'leverage',
     'executive',
     'branch',
     'spending',
     'power',
     'standing',
     'trump',
     'administration',
     'may',
     'come',
     'pressure',
     'advocate',
     'presidential',
     'authority',
     'fight',
     'house',
     'matter',
     'shared',
     'view',
     'health',
     'care',
     'since',
     'precedent',
     'could',
     'broad',
     'repercussion',
     'complicated',
     'set',
     'dynamic',
     'illustrating',
     'quick',
     'legal',
     'victory',
     'house',
     'trump',
     'era',
     'might',
     'come',
     'cost',
     'republican',
     'never',
     'anticipated',
     'took',
     'obama',
     'white',
     'house',
     'bullet',
     'shell',
     'get',
     'counted',
     'blood',
     'dry',
     'votive',
     'candle',
     'burn',
     'people',
     'peer',
     'window',
     'see',
     'crime',
     'scene',
     'gone',
     'cold',
     'band',
     'yellow',
     'police',
     'tape',
     'blowing',
     'breeze',
     'south',
     'bronx',
     'across',
     'harlem',
     'river',
     'manhattan',
     'shorthand',
     'urban',
     'dysfunction',
     'still',
     'suffers',
     'violence',
     'level',
     'long',
     'ago',
     'slashed',
     'many',
     'part',
     'new',
     'york',
     'city',
     'yet',
     'city',
     'effort',
     'fight',
     'remain',
     'splintered',
     'underfunded',
     'burdened',
     'scandal',
     'th',
     'precinct',
     'southern',
     'tip',
     'bronx',
     'poor',
     'minority',
     'neighborhood',
     'across',
     'country',
     'people',
     'long',
     'hounded',
     'infraction',
     'cry',
     'protection',
     'grievous',
     'injury',
     'death',
     'september',
     'four',
     'every',
     'five',
     'shooting',
     'precinct',
     'year',
     'unsolved',
     'city',
     'precinct',
     'th',
     'ha',
     'highest',
     'murder',
     'rate',
     'fewest',
     'detective',
     'per',
     'violent',
     'crime',
     'reflecting',
     'disparity',
     'staffing',
     'hit',
     'hardest',
     'neighborhood',
     'outside',
     'manhattan',
     'according',
     'new',
     'york',
     'time',
     'analysis',
     'police',
     'department',
     'data',
     'investigator',
     'precinct',
     'saddled',
     'twice',
     'number',
     'case',
     'department',
     'recommends',
     'even',
     'boss',
     'called',
     'police',
     'headquarters',
     'answer',
     'sharpest',
     'crime',
     'rise',
     'city',
     'year',
     'across',
     'bronx',
     'investigative',
     'resource',
     'squeezed',
     'ha',
     'highest',
     'rate',
     'city',
     'five',
     'borough',
     'thinnest',
     'detective',
     'staffing',
     'nine',
     'precinct',
     'detective',
     'squad',
     'violent',
     'crime',
     'city',
     'borough',
     'robbery',
     'squad',
     'smaller',
     'manhattan',
     'even',
     'though',
     'bronx',
     'ha',
     'case',
     'year',
     'homicide',
     'squad',
     'ha',
     'one',
     'detective',
     'every',
     'four',
     'murder',
     'compared',
     'one',
     'detective',
     'roughly',
     'every',
     'two',
     'murder',
     'upper',
     'manhattan',
     'one',
     'detective',
     'per',
     'murder',
     'lower',
     'manhattan',
     'lobby',
     'family',
     'apartment',
     'outside',
     'methadone',
     'clinic',
     'art',
     'studio',
     'people',
     'take',
     'note',
     'inequity',
     'hear',
     'police',
     'commander',
     'explain',
     'lack',
     'resource',
     'place',
     'floodlight',
     'dangerous',
     'block',
     'post',
     'officer',
     'corner',
     'watch',
     'witness',
     'cower',
     'behind',
     'door',
     'fearful',
     'gunman',
     'crew',
     'confident',
     'police',
     'department',
     'ability',
     'protect',
     'though',
     'people',
     'see',
     'lot',
     'rarely',
     'testify',
     'south',
     'bronx',
     'many',
     'predominantly',
     'black',
     'hispanic',
     'neighborhood',
     'like',
     'united',
     'state',
     'contract',
     'police',
     'community',
     'tatter',
     'people',
     'story',
     'crime',
     'report',
     'ignored',
     'call',
     'went',
     'unanswered',
     'hour',
     'others',
     'tell',
     'call',
     'help',
     'ending',
     'caller',
     'arrest',
     'minor',
     'charge',
     'leading',
     'hour',
     'fetid',
     'holding',
     'cell',
     'paradox',
     'policing',
     'th',
     'precinct',
     'neighborhood',
     'historically',
     'prime',
     'target',
     'aggressive',
     'tactic',
     'like',
     'designed',
     'ward',
     'disorder',
     'precinct',
     'detective',
     'le',
     'time',
     'anywhere',
     'else',
     'city',
     'answer',
     'blood',
     'spilled',
     'violent',
     'crime',
     'gola',
     'white',
     'wa',
     'beside',
     'daughter',
     'wa',
     'shot',
     'killed',
     'playground',
     'summer',
     'four',
     'year',
     'son',
     'wa',
     'gunned',
     'housing',
     'project',
     'ticked',
     'public',
     'safety',
     'resource',
     'said',
     'scant',
     'bronx',
     'neighborhood',
     'like',
     'security',
     'camera',
     'light',
     'lock',
     'investigating',
     'police',
     'officer',
     'nothing',
     'said',
     'come',
     'family',
     'said',
     'authority',
     'really',
     'care',
     'much',
     'feel',
     'time',
     'ha',
     'documenting',
     'murder',
     'logged',
     'year',
     'th',
     'precinct',
     'one',
     'handful',
     'neighborhood',
     'deadly',
     'violence',
     'remains',
     'problem',
     'era',
     'crime',
     'new',
     'york',
     'city',
     'homicide',
     'precinct',
     'year',
     'nine',
     'strain',
     'detective',
     'go',
     'unsolved',
     'half',
     'year',
     'look',
     'take',
     'law',
     'hand',
     'hundred',
     'conversation',
     'grieving',
     'relative',
     'friend',
     'witness',
     'police',
     'officer',
     'social',
     'force',
     'flare',
     'murder',
     'place',
     'like',
     'th',
     'precinct',
     'become',
     'clearer',
     'merciless',
     'gang',
     'code',
     'mental',
     'illness',
     'drug',
     'long',
     'memory',
     'feud',
     'simmered',
     'officer',
     'view',
     'reason',
     'murder',
     'never',
     'solved',
     'also',
     'emerge',
     'paralyzing',
     'fear',
     'retribution',
     'victim',
     'carrying',
     'secret',
     'graf',
     'relentless',
     'casework',
     'force',
     'detective',
     'move',
     'hope',
     'break',
     'come',
     'later',
     'frustration',
     'build',
     'side',
     'detective',
     'phone',
     'rarely',
     'ring',
     'tip',
     'officer',
     'grow',
     'embittered',
     'witness',
     'cooperate',
     'meantime',
     'victim',
     'friend',
     'conduct',
     'investigation',
     'talk',
     'grabbing',
     'stash',
     'gun',
     'wheel',
     'well',
     'mother',
     'apartment',
     'find',
     'suspect',
     'chasm',
     'police',
     'community',
     'gang',
     'gun',
     'violence',
     'flourish',
     'parent',
     'try',
     'protect',
     'family',
     'drug',
     'crew',
     'threat',
     'officer',
     'work',
     'overcome',
     'residue',
     'year',
     'mistrust',
     'understaffing',
     'community',
     'still',
     'go',
     'racing',
     'one',
     'call',
     'next',
     'street',
     'around',
     'st',
     'mary',
     'park',
     'scene',
     'two',
     'fatal',
     'shooting',
     'logged',
     'th',
     'precinct',
     'year',
     'unsolved',
     'james',
     'fernandez',
     'heard',
     'talk',
     ...]




```python
#df2 = df2.map(lambda x:word_lemmatizer(x) ).map(lambda x:word_stem(x) )
 
#stem =  lambda x: word_stem(x)  
#lemma = lambda x: word_lemmatizer(x)  
#K = lemma(stem(df3))
#len(  flatten( K)) 
```


```python
# Remove Duplcate and sort to list
vocab = list(sorted(set(vocab))) 
```


```python
vocab
```




    ['aaron',
     'abadi',
     'abandon',
     'abandoned',
     'abates',
     'abbas',
     'abc',
     'abdulkarim',
     'abe',
     'ability',
     'able',
     'abnormal',
     'aborted',
     'absence',
     'absent',
     'absorb',
     'absurd',
     'abundant',
     'abuse',
     'abusive',
     'academy',
     'accelerate',
     'accelerated',
     'acceptable',
     'acceptance',
     'accepted',
     'access',
     'accessible',
     'accessory',
     'accidentally',
     'acclaim',
     'accommodate',
     'accomplishment',
     'accordance',
     'according',
     'accordingly',
     'account',
     'accounting',
     'accurate',
     'achieve',
     'achievement',
     'acknowledge',
     'acknowledged',
     'acknowledging',
     'acolyte',
     'acra',
     'acronym',
     'across',
     'acrylic',
     'act',
     'acted',
     'acting',
     'action',
     'active',
     'activist',
     'activity',
     'actor',
     'actress',
     'actual',
     'actually',
     'acute',
     'acutely',
     'ad',
     'adam',
     'adamantly',
     'adapting',
     'add',
     'added',
     'addendum',
     'addict',
     'adding',
     'addition',
     'additional',
     'address',
     'adequate',
     'adhere',
     'adjust',
     'adjustment',
     'adm',
     'administration',
     'admiral',
     'admired',
     'admission',
     'admit',
     'admits',
     'adopt',
     'adorama',
     'adorning',
     'adult',
     'adulyadej',
     'advance',
     'advanced',
     'advancing',
     'advantage',
     'adventure',
     'advice',
     'advised',
     'adviser',
     'advises',
     'advising',
     'advocate',
     'advocated',
     'aerial',
     'aesthetic',
     'affair',
     'affect',
     'affected',
     'affecting',
     'affinity',
     'afflicts',
     'affluent',
     'afford',
     'affordable',
     'affront',
     'afghan',
     'afghanistan',
     'afraid',
     'afresh',
     'africa',
     'african',
     'afterglow',
     'afternoon',
     'afterward',
     'age',
     'aged',
     'agency',
     'agenda',
     'agent',
     'aggravating',
     'aggressive',
     'aggressively',
     'aging',
     'ago',
     'agony',
     'agree',
     'agreed',
     'agreement',
     'agriculture',
     'ahead',
     'ahmad',
     'aid',
     'aide',
     'aim',
     'aimed',
     'aiming',
     'air',
     'aircraft',
     'aired',
     'airplane',
     'airport',
     'airworthy',
     'aisle',
     'aiymkan',
     'ajayi',
     'akira',
     'al',
     'alabama',
     'alan',
     'albee',
     'album',
     'aleppo',
     'alerted',
     'alexander',
     'algaier',
     'ali',
     'alienating',
     'align',
     'alive',
     'allam',
     'alleged',
     'allegiance',
     'allison',
     'allocated',
     'allow',
     'allowed',
     'allowing',
     'allows',
     'ally',
     'almost',
     'alone',
     'along',
     'alongside',
     'aloof',
     'already',
     'also',
     'altered',
     'although',
     'alto',
     'alum',
     'aluminum',
     'alvin',
     'always',
     'alzheimer',
     'amateur',
     'amazing',
     'ambassador',
     'ambiguous',
     'ambitious',
     'ambulance',
     'ambush',
     'amendment',
     'amenity',
     'america',
     'american',
     'americanized',
     'amid',
     'amine',
     'among',
     'amount',
     'anachronism',
     'anadolu',
     'analysis',
     'analyst',
     'analyzed',
     'analyzing',
     'ancient',
     'andrew',
     'andrey',
     'androgynous',
     'andrzej',
     'andy',
     'angel',
     'angeles',
     'angering',
     'angry',
     'animal',
     'animate',
     'animated',
     'animation',
     'animator',
     'anita',
     'ankara',
     'anne',
     'annette',
     'announced',
     'announcement',
     'annoying',
     'annual',
     'anonymity',
     'another',
     'answer',
     'answered',
     'answering',
     'antagonized',
     'antagonizing',
     'anthony',
     'anthropology',
     'anticipated',
     'anticipating',
     'anticorruption',
     'antiquity',
     'antiwar',
     'antonin',
     'antonoff',
     'anybody',
     'anymore',
     'anyone',
     'anything',
     'anyway',
     'anywhere',
     'apart',
     'apartment',
     'aperture',
     'apology',
     'app',
     'apparent',
     'apparently',
     'appeal',
     'appealed',
     'appear',
     'appearance',
     'appeared',
     'appears',
     'appellate',
     'apple',
     'application',
     'applied',
     'apply',
     'applying',
     'appoint',
     'appointed',
     'appointee',
     'appoints',
     'appreciate',
     'approach',
     'approached',
     'appropriate',
     'appropriated',
     'appropriating',
     'appropriation',
     'approval',
     'approve',
     'april',
     'arab',
     'arc',
     'arch',
     'archaeological',
     'archaeologist',
     'archaeology',
     'archdiocese',
     'architect',
     'architectural',
     'ardent',
     'arduously',
     'area',
     'arena',
     'argenis',
     'argue',
     'argued',
     'argument',
     'arise',
     'arm',
     'armed',
     'arming',
     'army',
     'arnie',
     'arnold',
     'arose',
     'around',
     'arranged',
     'arrest',
     'arrested',
     'arrival',
     'arrived',
     'art',
     'artful',
     'arthur',
     'article',
     'articulated',
     'artifact',
     'artist',
     'artistic',
     'artwork',
     'asaad',
     'ascended',
     'ascension',
     'ash',
     'ashraf',
     'asia',
     'asiabriefingnytimes',
     'asian',
     'ask',
     'asked',
     'asking',
     'asparagus',
     'aspect',
     'aspirant',
     'ass',
     'assailant',
     'assassinated',
     'assault',
     'assemble',
     'assembly',
     'assert',
     'asserted',
     'assertion',
     'assessment',
     'assigned',
     'assignment',
     'assigns',
     'assistant',
     'assisting',
     'associate',
     'associated',
     'association',
     'assume',
     'assumed',
     'assured',
     'assyrian',
     'asterisk',
     'astonishing',
     'astoria',
     'astrophysics',
     'athlete',
     'athletic',
     'athletically',
     'atkin',
     'atlanta',
     'atmosphere',
     'atmospheric',
     'attached',
     'attack',
     'attacked',
     'attempt',
     'attempted',
     'attempting',
     'attend',
     'attendance',
     'attendant',
     'attended',
     'attendee',
     'attending',
     'attention',
     'attitude',
     'attorney',
     'attracted',
     'attributed',
     'attuned',
     'auction',
     'audacious',
     'audience',
     'audio',
     'audrey',
     'august',
     'auld',
     'aunt',
     'aura',
     'auschwitz',
     'australia',
     'austrian',
     'auteur',
     'author',
     'authority',
     'authorized',
     'autobahn',
     'automaking',
     'automatic',
     'autopsy',
     'ava',
     'available',
     'avatar',
     'avenue',
     'average',
     'avoid',
     'avoided',
     'await',
     'aware',
     'away',
     'awkward',
     'awning',
     'ax',
     'b',
     'back',
     'backed',
     'backfire',
     'background',
     'backlash',
     'backpack',
     'backup',
     'backward',
     'bad',
     'badger',
     'badly',
     'bag',
     'baggage',
     'baghdad',
     'bagram',
     'bakery',
     'balance',
     'ball',
     'ballistic',
     'bambi',
     'bamiyan',
     'ban',
     'band',
     'banging',
     'bank',
     'banned',
     'banquette',
     'banter',
     'bar',
     'barack',
     'barbieri',
     'barely',
     'bargaining',
     'bariatric',
     'barney',
     'baroque',
     'barrack',
     'barred',
     'barrel',
     'barring',
     'barron',
     'base',
     'baseball',
     'based',
     'basement',
     'bashar',
     'basic',
     'basically',
     'basis',
     'basketball',
     'batch',
     'bathroom',
     'battle',
     'battled',
     'battling',
     'bay',
     'bayside',
     'beach',
     'bear',
     'beat',
     'beaten',
     'beating',
     'beatle',
     'beautiful',
     'beautifully',
     'beauty',
     'became',
     'become',
     'becomes',
     'becoming',
     'bed',
     'bedbedbedbedbed',
     'bedeviled',
     'bedford',
     'bedsheet',
     'beer',
     'began',
     'begin',
     'begun',
     'behalf',
     'behar',
     'behind',
     'beijing',
     'belatedly',
     'belie',
     'belief',
     'believe',
     'believed',
     'bell',
     'belong',
     'belonged',
     'belting',
     'belz',
     'ben',
     'bench',
     'benching',
     'bending',
     'beneath',
     'benefit',
     'benjamin',
     'benji',
     'bent',
     'bergman',
     'bernstein',
     'berrigan',
     'berth',
     'beside',
     'besides',
     'best',
     'bestowed',
     'bet',
     'betances',
     'betsy',
     'better',
     'bewildering',
     'beyond',
     'bhalla',
     'bharatiya',
     'bhumibol',
     'bias',
     'bible',
     'bicycle',
     'bid',
     'bidder',
     'bidding',
     'biden',
     'big',
     'bigger',
     'biggest',
     'bike',
     'bill',
     'billion',
     'billionaire',
     'bin',
     'bindal',
     'binge',
     'biographer',
     'biological',
     'biology',
     'bipartisan',
     'birch',
     'bird',
     'birth',
     'birthplace',
     'bit',
     'bite',
     'bitter',
     'bitterness',
     'black',
     'blame',
     'blamed',
     'blaming',
     'blando',
     'blanket',
     'blasio',
     'blast',
     'blazer',
     'bleacher',
     'bleak',
     'bleakest',
     'blending',
     'blessed',
     'blew',
     'blizzard',
     'block',
     'blockbuster',
     'blog',
     'blond',
     'blood',
     'bloody',
     'bloomberg',
     'blow',
     'blowing',
     'blown',
     'blue',
     'bluegrass',
     'blume',
     'bluntly',
     'blurry',
     'board',
     'boardinghouse',
     'bob',
     'bobby',
     'body',
     'boehner',
     'bohemian',
     'boiler',
     'bold',
     'bolshie',
     'bolster',
     'bomb',
     'bombastic',
     'bombed',
     'bomber',
     'bombing',
     'bond',
     'bone',
     'bonnie',
     'bono',
     'book',
     'booked',
     'boom',
     'booming',
     'boost',
     'boot',
     'boozy',
     'border',
     'born',
     'borough',
     'bos',
     'bosnia',
     'boss',
     'boston',
     'bothered',
     'bottle',
     'bought',
     'boulevard',
     'boulez',
     'boulud',
     'bouncing',
     'bound',
     'boutros',
     'bow',
     'bowie',
     'bowl',
     'box',
     'boxer',
     'boxshall',
     'boxy',
     'boy',
     'boyce',
     'boyfriend',
     'brady',
     'bragging',
     'brain',
     'braith',
     'braithophone',
     'brake',
     'branca',
     'branch',
     'brand',
     'brave',
     'brawl',
     'brazenly',
     'brazil',
     'break',
     'breakfast',
     'breaking',
     'breast',
     'breathe',
     'breeze',
     'brian',
     'bribe',
     'bribery',
     'brick',
     'bride',
     'bridge',
     'brief',
     'briefing',
     'briefly',
     'brigade',
     'bright',
     'brilliant',
     'bring',
     'bringing',
     'brings',
     'brinkley',
     'brinkmanship',
     'brisk',
     'britain',
     'british',
     'broad',
     'broadcast',
     'broadened',
     'broader',
     'broadly',
     'broadway',
     'broccoli',
     'brock',
     'broken',
     'bronco',
     'bronx',
     'brooding',
     'brookings',
     'brooklyn',
     'brookner',
     'brothel',
     'brother',
     'brought',
     'brown',
     'brownstein',
     'bruce',
     'bruck',
     'bruised',
     'brunch',
     'brush',
     'brutal',
     'brzezinski',
     'buccaneer',
     'buck',
     'buckingham',
     'budget',
     'buffet',
     'bug',
     'build',
     'building',
     'buildup',
     'built',
     'bulb',
     'bulk',
     'bull',
     'bullet',
     'bulletproof',
     'bulochnikov',
     'bunch',
     'burden',
     'burdened',
     'bureau',
     'bureaucratic',
     'burgundy',
     'burke',
     'burlap',
     'burn',
     'burned',
     'burner',
     'burning',
     'burnish',
     'burst',
     'bus',
     'bush',
     'busiest',
     'business',
     'businessman',
     'bustier',
     'busting',
     'busy',
     'butcher',
     'butterfly',
     'butterlin',
     'button',
     'buy',
     'buyer',
     'buying',
     'bye',
     'c',
     'cab',
     'cabana',
     'cabinet',
     'cable',
     'cache',
     'cadre',
     'cafeteria',
     'cahill',
     'cajoled',
     'cake',
     'calculation',
     'calendar',
     'calif',
     'california',
     'caliphate',
     'call',
     'called',
     'caller',
     'calligrapher',
     'calligraphy',
     'calling',
     'calm',
     'calorie',
     'came',
     'camellia',
     'camera',
     'camilla',
     'camouflaged',
     'camp',
     'campaign',
     'campaigned',
     'campaigning',
     'campbell',
     'canagliflozin',
     'canceled',
     'cancer',
     'candidate',
     'candis',
     'candle',
     'candy',
     'canemaker',
     'cannot',
     'canvas',
     'capable',
     'capacity',
     'capital',
     'capitalized',
     'capitol',
     'capping',
     'capstone',
     'captive',
     'capture',
     'captured',
     'car',
     'carbon',
     'card',
     'care',
     'career',
     'carefully',
     'caregiver',
     'careless',
     'carey',
     'carlos',
     'carol',
     'carole',
     'carolina',
     'carpet',
     'carping',
     'carr',
     'carrie',
     'carried',
     'carrier',
     'carry',
     'carrying',
     'carted',
     'cascading',
     'case',
     'caseloads',
     'casework',
     'cash',
     'cashless',
     'cast',
     'castigated',
     'castle',
     'castro',
     'casual',
     'casualty',
     'cat',
     'categorized',
     'catholic',
     'catwalk',
     'caucus',
     'caught',
     'cause',
     'caused',
     'causing',
     'cautioned',
     'cave',
     'cayne',
     'cbi',
     'cbs',
     'cd',
     'ceiling',
     'celebrant',
     'celebrated',
     'celebration',
     'celebrity',
     'cell',
     'cellphone',
     'cemented',
     'cent',
     'centenary',
     'center',
     'centered',
     'centerpiece',
     'centipede',
     'central',
     'centralized',
     'century',
     'ceramic',
     'cereal',
     'ceremony',
     'certain',
     'certainly',
     'cesspool',
     'cetera',
     'chafed',
     'chain',
     'chair',
     'chairman',
     'chaise',
     'challenge',
     'chamber',
     'champion',
     'championed',
     'championship',
     'chance',
     'chandelier',
     'change',
     'changed',
     'changer',
     'changing',
     'channel',
     'chaos',
     'character',
     'chard',
     'charge',
     'charged',
     'charger',
     'charity',
     'charles',
     'charleston',
     'charlotte',
     'chasing',
     'chasm',
     'chastened',
     'chatting',
     'cheap',
     'cheat',
     'cheated',
     'chechi',
     'check',
     'checking',
     'checkpoint',
     'cheer',
     'cheered',
     'chef',
     'cheikhmous',
     'chemical',
     'chen',
     'cheong',
     'chester',
     'chic',
     'chicago',
     'chicken',
     'chief',
     'child',
     'china',
     'chinese',
     'chip',
     'chocolate',
     'choi',
     'choice',
     'choir',
     'choked',
     'choose',
     'choosing',
     'choreography',
     'chorus',
     'chose',
     'chosen',
     'chris',
     'christened',
     'christian',
     'christmas',
     'christy',
     'chronic',
     'chronicle',
     'chuck',
     'church',
     'churchgoing',
     'cigarette',
     'cindy',
     'circle',
     'circuit',
     'circulated',
     'circulating',
     'circulation',
     'circumstance',
     'circumvent',
     'circumventing',
     'citadel',
     'cited',
     'citizen',
     'city',
     'citywide',
     'civil',
     'civilian',
     'clad',
     'claim',
     'claimed',
     'clandestine',
     'clarity',
     ...]




```python
#  16179 uniq , 3662 unique char ,24638 
print ('{} unique characters'.format(len(vocab)))
```

    6381 unique characters
    


```python
vocab
```




    ['aaron',
     'abadi',
     'abandon',
     'abandoned',
     'abates',
     'abbas',
     'abc',
     'abdulkarim',
     'abe',
     'ability',
     'able',
     'abnormal',
     'aborted',
     'absence',
     'absent',
     'absorb',
     'absurd',
     'abundant',
     'abuse',
     'abusive',
     'academy',
     'accelerate',
     'accelerated',
     'acceptable',
     'acceptance',
     'accepted',
     'access',
     'accessible',
     'accessory',
     'accidentally',
     'acclaim',
     'accommodate',
     'accomplishment',
     'accordance',
     'according',
     'accordingly',
     'account',
     'accounting',
     'accurate',
     'achieve',
     'achievement',
     'acknowledge',
     'acknowledged',
     'acknowledging',
     'acolyte',
     'acra',
     'acronym',
     'across',
     'acrylic',
     'act',
     'acted',
     'acting',
     'action',
     'active',
     'activist',
     'activity',
     'actor',
     'actress',
     'actual',
     'actually',
     'acute',
     'acutely',
     'ad',
     'adam',
     'adamantly',
     'adapting',
     'add',
     'added',
     'addendum',
     'addict',
     'adding',
     'addition',
     'additional',
     'address',
     'adequate',
     'adhere',
     'adjust',
     'adjustment',
     'adm',
     'administration',
     'admiral',
     'admired',
     'admission',
     'admit',
     'admits',
     'adopt',
     'adorama',
     'adorning',
     'adult',
     'adulyadej',
     'advance',
     'advanced',
     'advancing',
     'advantage',
     'adventure',
     'advice',
     'advised',
     'adviser',
     'advises',
     'advising',
     'advocate',
     'advocated',
     'aerial',
     'aesthetic',
     'affair',
     'affect',
     'affected',
     'affecting',
     'affinity',
     'afflicts',
     'affluent',
     'afford',
     'affordable',
     'affront',
     'afghan',
     'afghanistan',
     'afraid',
     'afresh',
     'africa',
     'african',
     'afterglow',
     'afternoon',
     'afterward',
     'age',
     'aged',
     'agency',
     'agenda',
     'agent',
     'aggravating',
     'aggressive',
     'aggressively',
     'aging',
     'ago',
     'agony',
     'agree',
     'agreed',
     'agreement',
     'agriculture',
     'ahead',
     'ahmad',
     'aid',
     'aide',
     'aim',
     'aimed',
     'aiming',
     'air',
     'aircraft',
     'aired',
     'airplane',
     'airport',
     'airworthy',
     'aisle',
     'aiymkan',
     'ajayi',
     'akira',
     'al',
     'alabama',
     'alan',
     'albee',
     'album',
     'aleppo',
     'alerted',
     'alexander',
     'algaier',
     'ali',
     'alienating',
     'align',
     'alive',
     'allam',
     'alleged',
     'allegiance',
     'allison',
     'allocated',
     'allow',
     'allowed',
     'allowing',
     'allows',
     'ally',
     'almost',
     'alone',
     'along',
     'alongside',
     'aloof',
     'already',
     'also',
     'altered',
     'although',
     'alto',
     'alum',
     'aluminum',
     'alvin',
     'always',
     'alzheimer',
     'amateur',
     'amazing',
     'ambassador',
     'ambiguous',
     'ambitious',
     'ambulance',
     'ambush',
     'amendment',
     'amenity',
     'america',
     'american',
     'americanized',
     'amid',
     'amine',
     'among',
     'amount',
     'anachronism',
     'anadolu',
     'analysis',
     'analyst',
     'analyzed',
     'analyzing',
     'ancient',
     'andrew',
     'andrey',
     'androgynous',
     'andrzej',
     'andy',
     'angel',
     'angeles',
     'angering',
     'angry',
     'animal',
     'animate',
     'animated',
     'animation',
     'animator',
     'anita',
     'ankara',
     'anne',
     'annette',
     'announced',
     'announcement',
     'annoying',
     'annual',
     'anonymity',
     'another',
     'answer',
     'answered',
     'answering',
     'antagonized',
     'antagonizing',
     'anthony',
     'anthropology',
     'anticipated',
     'anticipating',
     'anticorruption',
     'antiquity',
     'antiwar',
     'antonin',
     'antonoff',
     'anybody',
     'anymore',
     'anyone',
     'anything',
     'anyway',
     'anywhere',
     'apart',
     'apartment',
     'aperture',
     'apology',
     'app',
     'apparent',
     'apparently',
     'appeal',
     'appealed',
     'appear',
     'appearance',
     'appeared',
     'appears',
     'appellate',
     'apple',
     'application',
     'applied',
     'apply',
     'applying',
     'appoint',
     'appointed',
     'appointee',
     'appoints',
     'appreciate',
     'approach',
     'approached',
     'appropriate',
     'appropriated',
     'appropriating',
     'appropriation',
     'approval',
     'approve',
     'april',
     'arab',
     'arc',
     'arch',
     'archaeological',
     'archaeologist',
     'archaeology',
     'archdiocese',
     'architect',
     'architectural',
     'ardent',
     'arduously',
     'area',
     'arena',
     'argenis',
     'argue',
     'argued',
     'argument',
     'arise',
     'arm',
     'armed',
     'arming',
     'army',
     'arnie',
     'arnold',
     'arose',
     'around',
     'arranged',
     'arrest',
     'arrested',
     'arrival',
     'arrived',
     'art',
     'artful',
     'arthur',
     'article',
     'articulated',
     'artifact',
     'artist',
     'artistic',
     'artwork',
     'asaad',
     'ascended',
     'ascension',
     'ash',
     'ashraf',
     'asia',
     'asiabriefingnytimes',
     'asian',
     'ask',
     'asked',
     'asking',
     'asparagus',
     'aspect',
     'aspirant',
     'ass',
     'assailant',
     'assassinated',
     'assault',
     'assemble',
     'assembly',
     'assert',
     'asserted',
     'assertion',
     'assessment',
     'assigned',
     'assignment',
     'assigns',
     'assistant',
     'assisting',
     'associate',
     'associated',
     'association',
     'assume',
     'assumed',
     'assured',
     'assyrian',
     'asterisk',
     'astonishing',
     'astoria',
     'astrophysics',
     'athlete',
     'athletic',
     'athletically',
     'atkin',
     'atlanta',
     'atmosphere',
     'atmospheric',
     'attached',
     'attack',
     'attacked',
     'attempt',
     'attempted',
     'attempting',
     'attend',
     'attendance',
     'attendant',
     'attended',
     'attendee',
     'attending',
     'attention',
     'attitude',
     'attorney',
     'attracted',
     'attributed',
     'attuned',
     'auction',
     'audacious',
     'audience',
     'audio',
     'audrey',
     'august',
     'auld',
     'aunt',
     'aura',
     'auschwitz',
     'australia',
     'austrian',
     'auteur',
     'author',
     'authority',
     'authorized',
     'autobahn',
     'automaking',
     'automatic',
     'autopsy',
     'ava',
     'available',
     'avatar',
     'avenue',
     'average',
     'avoid',
     'avoided',
     'await',
     'aware',
     'away',
     'awkward',
     'awning',
     'ax',
     'b',
     'back',
     'backed',
     'backfire',
     'background',
     'backlash',
     'backpack',
     'backup',
     'backward',
     'bad',
     'badger',
     'badly',
     'bag',
     'baggage',
     'baghdad',
     'bagram',
     'bakery',
     'balance',
     'ball',
     'ballistic',
     'bambi',
     'bamiyan',
     'ban',
     'band',
     'banging',
     'bank',
     'banned',
     'banquette',
     'banter',
     'bar',
     'barack',
     'barbieri',
     'barely',
     'bargaining',
     'bariatric',
     'barney',
     'baroque',
     'barrack',
     'barred',
     'barrel',
     'barring',
     'barron',
     'base',
     'baseball',
     'based',
     'basement',
     'bashar',
     'basic',
     'basically',
     'basis',
     'basketball',
     'batch',
     'bathroom',
     'battle',
     'battled',
     'battling',
     'bay',
     'bayside',
     'beach',
     'bear',
     'beat',
     'beaten',
     'beating',
     'beatle',
     'beautiful',
     'beautifully',
     'beauty',
     'became',
     'become',
     'becomes',
     'becoming',
     'bed',
     'bedbedbedbedbed',
     'bedeviled',
     'bedford',
     'bedsheet',
     'beer',
     'began',
     'begin',
     'begun',
     'behalf',
     'behar',
     'behind',
     'beijing',
     'belatedly',
     'belie',
     'belief',
     'believe',
     'believed',
     'bell',
     'belong',
     'belonged',
     'belting',
     'belz',
     'ben',
     'bench',
     'benching',
     'bending',
     'beneath',
     'benefit',
     'benjamin',
     'benji',
     'bent',
     'bergman',
     'bernstein',
     'berrigan',
     'berth',
     'beside',
     'besides',
     'best',
     'bestowed',
     'bet',
     'betances',
     'betsy',
     'better',
     'bewildering',
     'beyond',
     'bhalla',
     'bharatiya',
     'bhumibol',
     'bias',
     'bible',
     'bicycle',
     'bid',
     'bidder',
     'bidding',
     'biden',
     'big',
     'bigger',
     'biggest',
     'bike',
     'bill',
     'billion',
     'billionaire',
     'bin',
     'bindal',
     'binge',
     'biographer',
     'biological',
     'biology',
     'bipartisan',
     'birch',
     'bird',
     'birth',
     'birthplace',
     'bit',
     'bite',
     'bitter',
     'bitterness',
     'black',
     'blame',
     'blamed',
     'blaming',
     'blando',
     'blanket',
     'blasio',
     'blast',
     'blazer',
     'bleacher',
     'bleak',
     'bleakest',
     'blending',
     'blessed',
     'blew',
     'blizzard',
     'block',
     'blockbuster',
     'blog',
     'blond',
     'blood',
     'bloody',
     'bloomberg',
     'blow',
     'blowing',
     'blown',
     'blue',
     'bluegrass',
     'blume',
     'bluntly',
     'blurry',
     'board',
     'boardinghouse',
     'bob',
     'bobby',
     'body',
     'boehner',
     'bohemian',
     'boiler',
     'bold',
     'bolshie',
     'bolster',
     'bomb',
     'bombastic',
     'bombed',
     'bomber',
     'bombing',
     'bond',
     'bone',
     'bonnie',
     'bono',
     'book',
     'booked',
     'boom',
     'booming',
     'boost',
     'boot',
     'boozy',
     'border',
     'born',
     'borough',
     'bos',
     'bosnia',
     'boss',
     'boston',
     'bothered',
     'bottle',
     'bought',
     'boulevard',
     'boulez',
     'boulud',
     'bouncing',
     'bound',
     'boutros',
     'bow',
     'bowie',
     'bowl',
     'box',
     'boxer',
     'boxshall',
     'boxy',
     'boy',
     'boyce',
     'boyfriend',
     'brady',
     'bragging',
     'brain',
     'braith',
     'braithophone',
     'brake',
     'branca',
     'branch',
     'brand',
     'brave',
     'brawl',
     'brazenly',
     'brazil',
     'break',
     'breakfast',
     'breaking',
     'breast',
     'breathe',
     'breeze',
     'brian',
     'bribe',
     'bribery',
     'brick',
     'bride',
     'bridge',
     'brief',
     'briefing',
     'briefly',
     'brigade',
     'bright',
     'brilliant',
     'bring',
     'bringing',
     'brings',
     'brinkley',
     'brinkmanship',
     'brisk',
     'britain',
     'british',
     'broad',
     'broadcast',
     'broadened',
     'broader',
     'broadly',
     'broadway',
     'broccoli',
     'brock',
     'broken',
     'bronco',
     'bronx',
     'brooding',
     'brookings',
     'brooklyn',
     'brookner',
     'brothel',
     'brother',
     'brought',
     'brown',
     'brownstein',
     'bruce',
     'bruck',
     'bruised',
     'brunch',
     'brush',
     'brutal',
     'brzezinski',
     'buccaneer',
     'buck',
     'buckingham',
     'budget',
     'buffet',
     'bug',
     'build',
     'building',
     'buildup',
     'built',
     'bulb',
     'bulk',
     'bull',
     'bullet',
     'bulletproof',
     'bulochnikov',
     'bunch',
     'burden',
     'burdened',
     'bureau',
     'bureaucratic',
     'burgundy',
     'burke',
     'burlap',
     'burn',
     'burned',
     'burner',
     'burning',
     'burnish',
     'burst',
     'bus',
     'bush',
     'busiest',
     'business',
     'businessman',
     'bustier',
     'busting',
     'busy',
     'butcher',
     'butterfly',
     'butterlin',
     'button',
     'buy',
     'buyer',
     'buying',
     'bye',
     'c',
     'cab',
     'cabana',
     'cabinet',
     'cable',
     'cache',
     'cadre',
     'cafeteria',
     'cahill',
     'cajoled',
     'cake',
     'calculation',
     'calendar',
     'calif',
     'california',
     'caliphate',
     'call',
     'called',
     'caller',
     'calligrapher',
     'calligraphy',
     'calling',
     'calm',
     'calorie',
     'came',
     'camellia',
     'camera',
     'camilla',
     'camouflaged',
     'camp',
     'campaign',
     'campaigned',
     'campaigning',
     'campbell',
     'canagliflozin',
     'canceled',
     'cancer',
     'candidate',
     'candis',
     'candle',
     'candy',
     'canemaker',
     'cannot',
     'canvas',
     'capable',
     'capacity',
     'capital',
     'capitalized',
     'capitol',
     'capping',
     'capstone',
     'captive',
     'capture',
     'captured',
     'car',
     'carbon',
     'card',
     'care',
     'career',
     'carefully',
     'caregiver',
     'careless',
     'carey',
     'carlos',
     'carol',
     'carole',
     'carolina',
     'carpet',
     'carping',
     'carr',
     'carrie',
     'carried',
     'carrier',
     'carry',
     'carrying',
     'carted',
     'cascading',
     'case',
     'caseloads',
     'casework',
     'cash',
     'cashless',
     'cast',
     'castigated',
     'castle',
     'castro',
     'casual',
     'casualty',
     'cat',
     'categorized',
     'catholic',
     'catwalk',
     'caucus',
     'caught',
     'cause',
     'caused',
     'causing',
     'cautioned',
     'cave',
     'cayne',
     'cbi',
     'cbs',
     'cd',
     'ceiling',
     'celebrant',
     'celebrated',
     'celebration',
     'celebrity',
     'cell',
     'cellphone',
     'cemented',
     'cent',
     'centenary',
     'center',
     'centered',
     'centerpiece',
     'centipede',
     'central',
     'centralized',
     'century',
     'ceramic',
     'cereal',
     'ceremony',
     'certain',
     'certainly',
     'cesspool',
     'cetera',
     'chafed',
     'chain',
     'chair',
     'chairman',
     'chaise',
     'challenge',
     'chamber',
     'champion',
     'championed',
     'championship',
     'chance',
     'chandelier',
     'change',
     'changed',
     'changer',
     'changing',
     'channel',
     'chaos',
     'character',
     'chard',
     'charge',
     'charged',
     'charger',
     'charity',
     'charles',
     'charleston',
     'charlotte',
     'chasing',
     'chasm',
     'chastened',
     'chatting',
     'cheap',
     'cheat',
     'cheated',
     'chechi',
     'check',
     'checking',
     'checkpoint',
     'cheer',
     'cheered',
     'chef',
     'cheikhmous',
     'chemical',
     'chen',
     'cheong',
     'chester',
     'chic',
     'chicago',
     'chicken',
     'chief',
     'child',
     'china',
     'chinese',
     'chip',
     'chocolate',
     'choi',
     'choice',
     'choir',
     'choked',
     'choose',
     'choosing',
     'choreography',
     'chorus',
     'chose',
     'chosen',
     'chris',
     'christened',
     'christian',
     'christmas',
     'christy',
     'chronic',
     'chronicle',
     'chuck',
     'church',
     'churchgoing',
     'cigarette',
     'cindy',
     'circle',
     'circuit',
     'circulated',
     'circulating',
     'circulation',
     'circumstance',
     'circumvent',
     'circumventing',
     'citadel',
     'cited',
     'citizen',
     'city',
     'citywide',
     'civil',
     'civilian',
     'clad',
     'claim',
     'claimed',
     'clandestine',
     'clarity',
     ...]




```python
df2
```




    0     [washington, congressional, republican, new, f...
    1     [bullet, shell, get, counted, blood, dry, voti...
    2     [walt, disney, bambi, opened, critic, praised,...
    3     [death, may, great, equalizer, necessarily, ev...
    4     [seoul, south, korea, north, korea, leader, ki...
    5     [london, queen, elizabeth, ii, ha, battling, c...
    6     [beijing, president, tsai, taiwan, sharply, cr...
    7     [danny, cahill, stood, slightly, dazed, blizza...
    8     [hillary, kerr, founder, digital, medium, comp...
    9     [angel, everywhere, mu, iz, family, apartment,...
    10    [donald, j, trump, take, control, white, house...
    11    [thompson, tex, one, promising, troubled, tech...
    12    [west, palm, beach, fla, donald, j, trump, ran...
    13    [article, part, series, aimed, helping, naviga...
    14    [season, family, travel, photo, perhaps, enlar...
    15    [finally, second, avenue, subway, opened, new,...
    16    [page, journal, found, dylann, roof, car, asse...
    17    [mumbai, india, wa, bold, risky, gamble, prime...
    18    [baghdad, suicide, bomber, detonated, pickup, ...
    19    [sydney, australia, annual, beach, pilgrimage,...
    20    [green, bay, packer, lost, washington, redskin...
    21    [mariah, carey, suffered, performance, train, ...
    22    [paris, islamic, state, wa, driven, ancient, c...
    23    [pop, music, fashion, never, met, cuter, georg...
    24    [washington, powerful, ambitious, congress, ye...
    25    [washington, time, republican, tumultuous, dec...
    26    [good, morning, need, know, turkish, authority...
    27    [body, iraqi, prisoner, wa, found, naked, badl...
    28    [istanbul, islamic, state, monday, issued, rar...
    29    [washington, president, obama, adviser, wrestl...
    Name: content, dtype: object




```python
#Flat list 
df2 =flatten(list(df2))
```

## Map vocab to indices and idx to vocab


```python
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in df2 ])
```


```python
idx2char[12] # test
```




    'aborted'




```python
# Show how the first 13 words from the text are mapped to integers
print ('{} ---- characters mapped to int ---- > {}'.format(repr(df2[:13]), text_as_int[:13]))
```

    ['washington', 'congressional', 'republican', 'new', 'fear', 'come', 'health', 'care', 'lawsuit', 'obama', 'administration', 'might', 'win'] ---- characters mapped to int ---- > [6199 1175 4760 3787 2107 1083 2607  833 3214 3860   79 3578 6270]
    


```python
seq_length = 100
examples_per_epoch = len(df2)//seq_length
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
```


```python
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)
```


```python
# Batch size
BATCH_SIZE = 12
#BATCH_SIZE = 64
steps_per_epoch = examples_per_epoch//BATCH_SIZE

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

dataset
```




    <BatchDataset shapes: ((12, 100), (12, 100)), types: (tf.int32, tf.int32)>




```python
# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024
```


```python
if tf.test.is_gpu_available():
  rnn = tf.keras.layers.LSTM
else:
  import functools
  rnn = functools.partial(
    tf.keras.layers.GRU, recurrent_activation='sigmoid')
```


```python
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(embedding_dim, return_sequences=True),
    rnn(rnn_units,
        return_sequences=True,
        recurrent_initializer='glorot_uniform',
        stateful=True),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model
```


```python
model = build_model(
  vocab_size = len(vocab),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)
optimizer = tf.optimizers.Adam()

model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding (Embedding)        (12, None, 256)           1633536   
    _________________________________________________________________
    gru (GRU)                    (12, None, 256)           394752    
    _________________________________________________________________
    lstm (LSTM)                  (12, None, 1024)          5246976   
    _________________________________________________________________
    dense (Dense)                (12, None, 6381)          6540525   
    =================================================================
    Total params: 13,815,789
    Trainable params: 13,815,789
    Non-trainable params: 0
    _________________________________________________________________
    


```python
# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)
model = build_model(
  vocab_size = len(vocab),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)
optimizer = tf.optimizers.Adam()
# Training step
EPOCHS = 100

```


```python

for epoch in range(EPOCHS):
    start = time.time()

    # initializing the hidden state at the start of every epoch
    # initially hidden is None
    hidden = model.reset_states()

    for (batch_n, (inp, target)) in enumerate(dataset):
          with tf.GradientTape() as tape:
              # feeding the hidden state back into the model
              # This is the interesting step
              predictions = model(inp)
               
              loss = tf.nn.sparse_softmax_cross_entropy_with_logits(target, predictions)

          grads = tape.gradient(loss, model.trainable_variables)
          optimizer.apply_gradients(zip(grads, model.trainable_variables))

          if batch_n % 100 == 0:
               pass
               # print('Epoch {0} Batch {1} Loss:{2}'.format(epoch+1, batch_n, tf.print(loss)[0]))
                
               

    # saving (checkpoint) the model every 5 epochs
    if (epoch ) % 10 == 0:
      model.save_weights(checkpoint_prefix.format(epoch=epoch))
    if (epoch ) % 1000 == 0:
      model.save_weights(checkpoint_prefix.format(epoch=epoch))    
    print ('Epoch {} Loss {}'.format(epoch+1, tf.print(loss) ))
    print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

model.save_weights(checkpoint_prefix.format(epoch=epoch))
```

    [[6.54943657 10.7108364 7.70784 ... 10.5168762 10.3693361 10.2134657]
     [10.002512 6.20749664 9.56113434 ... 7.54655838 6.7380805 6.65161896]
     [8.00521469 7.10256767 8.53765 ... 7.77186108 10.6190348 10.6420145]
     ...
     [8.27437592 10.8157864 7.69519615 ... 8.25664902 7.27781677 8.38977528]
     [7.05976677 10.6961632 10.5443811 ... 10.7168694 8.11445618 7.21030235]
     [8.41799259 10.861969 8.42685413 ... 7.23591423 7.75702572 6.23599911]]
    Epoch 1 Loss None
    Time taken for 1 epoch 4.733989715576172 sec
    
    [[6.38502264 10.4948711 7.41504908 ... 9.99693871 9.85391808 9.85525894]
     [9.59857 5.60096931 8.7842617 ... 6.62941742 6.49822521 5.96926]
     [7.98995 6.47586918 8.18765068 ... 7.66451359 10.5180912 10.5307169]
     ...
     [8.28011703 10.1315403 7.50955772 ... 8.30326366 6.61778641 8.43718]
     [6.92724 10.4692974 10.4692974 ... 10.4679127 7.52141476 7.25957537]
     [8.26479244 10.5201693 8.61718082 ... 6.81174946 7.80237 5.61931896]]
    Epoch 2 Loss None
    Time taken for 1 epoch 3.1337075233459473 sec
    
    [[6.3332324 10.2064543 7.51156235 ... 9.62806511 9.50024 9.53492355]
     [9.20524502 5.28056574 8.52995682 ... 6.66992569 6.63518238 5.9740324]
     [8.23292923 6.49417067 8.08384132 ... 7.63012123 10.229867 10.2425919]
     ...
     [8.28210258 9.69117451 7.49928 ... 8.38876629 6.51945257 8.45837402]
     [7.04122543 10.2126913 10.235321 ... 10.2238235 7.51244736 7.28079414]
     [8.3325119 10.230032 8.68156624 ... 6.80552959 7.7448349 5.24944878]]
    Epoch 3 Loss None
    Time taken for 1 epoch 3.1246578693389893 sec
    
    [[6.36665 9.88697243 7.52587652 ... 9.36111832 9.09340096 9.30900574]
     [9.78973579 5.93543148 9.56559372 ... 6.5366478 6.30975723 5.64308834]
     [8.27002144 6.01263332 8.03316784 ... 7.62078 9.45541573 9.846035]
     ...
     [8.23178768 9.24063301 7.56315136 ... 8.34588623 6.65375376 8.66479874]
     [6.83358 9.78405476 9.87076378 ... 9.70146942 7.51383972 7.36046457]
     [8.30090618 9.70031261 8.57884598 ... 6.67872143 7.76128101 5.47212934]]
    Epoch 4 Loss None
    Time taken for 1 epoch 3.1675517559051514 sec
    
    [[6.26287842 9.97480774 7.44746113 ... 9.27300644 8.79001331 8.98337841]
     [10.5626011 6.31949 9.53264523 ... 6.22403336 6.42517471 5.11134148]
     [8.09025478 6.02255249 8.07231 ... 7.551682 9.13280773 9.1185627]
     ...
     [8.21436882 8.9228 7.47413778 ... 8.28591919 7.14516973 8.62022209]
     [6.85795784 9.73744774 9.30483723 ... 9.96840572 7.45958519 6.96101809]
     [8.3764782 9.26722431 8.49971771 ... 6.72042513 7.60711861 6.10081959]]
    Epoch 5 Loss None
    Time taken for 1 epoch 3.1440391540527344 sec
    
    [[6.3684783 10.4737139 7.26492405 ... 9.33412075 8.80966759 8.83269787]
     [10.8172588 6.88395786 9.22315693 ... 6.43963814 6.44372654 4.92930031]
     [7.94244909 6.3253336 8.1072607 ... 7.648417 9.13436699 8.93635941]
     ...
     [8.12631226 8.87999 7.47956181 ... 8.31241608 6.70614529 8.86553764]
     [6.79548 9.66776085 9.12849808 ... 9.77191353 7.46317816 6.80529308]
     [8.52508545 9.14408875 8.4899416 ... 6.46475029 7.74478483 6.17313433]]
    Epoch 6 Loss None
    Time taken for 1 epoch 3.1377387046813965 sec
    
    [[6.39157248 10.5427523 7.44602108 ... 8.38350296 9.11191 7.98548508]
     [11.0910139 7.10856247 9.70467949 ... 6.67773342 5.85207701 4.89533424]
     [7.60747051 6.50112438 8.09328938 ... 7.23173141 9.28907585 8.62094593]
     ...
     [7.9142046 8.86087227 7.66595173 ... 8.37417603 6.2566328 8.47990608]
     [6.81412268 9.40738297 9.07802773 ... 9.86114311 7.4112854 6.81624222]
     [8.69950485 9.02669907 8.33013535 ... 6.47464275 7.29526472 6.5275259]]
    Epoch 7 Loss None
    Time taken for 1 epoch 3.1347110271453857 sec
    
    [[6.20162678 9.6117 7.30425739 ... 7.8820343 9.23723412 7.54914093]
     [9.87213612 7.15753746 8.52218342 ... 7.10420895 6.53768539 4.87788391]
     [7.5067234 6.4353981 7.8789649 ... 7.23564386 9.84854507 8.40968]
     ...
     [7.72523642 8.86336327 7.92802715 ... 8.48848057 6.17695236 8.33380222]
     [6.80952835 9.03209877 9.05324745 ... 9.09880066 7.30588818 6.60583591]
     [8.60681629 9.06658363 8.00684166 ... 6.20716858 7.16954422 6.87252]]
    Epoch 8 Loss None
    Time taken for 1 epoch 3.135708808898926 sec
    
    [[6.17789936 9.64016914 7.00190258 ... 7.46722031 8.60763264 7.7087512]
     [9.11849689 6.64524937 8.6469574 ... 7.1762085 6.59022522 4.28537798]
     [7.20603561 6.46503162 7.96323586 ... 6.81953096 9.52330399 8.3132534]
     ...
     [7.35510778 8.89639664 7.58652878 ... 8.20448 5.94684315 8.09471416]
     [7.54539 8.90023613 8.80086136 ... 9.56119919 7.13257122 6.47957659]
     [8.52947521 9.14555836 7.76175404 ... 5.60183144 6.79189205 6.18330193]]
    Epoch 9 Loss None
    Time taken for 1 epoch 3.1762423515319824 sec
    
    [[6.80593872 10.309166 6.60769415 ... 6.67843819 8.49218845 7.92638445]
     [8.17455769 7.105968 7.98317289 ... 7.19493437 5.84185648 4.52511454]
     [7.18199539 6.65277529 8.28982067 ... 6.42498493 9.01506424 8.26557732]
     ...
     [7.34130478 9.28650856 6.89104271 ... 7.82086754 6.09433937 7.51120186]
     [7.42701864 8.7013607 8.51362228 ... 8.7809248 5.87281322 5.88216305]
     [8.31199455 9.08353615 7.60173845 ... 5.29524136 6.63227224 6.33350039]]
    Epoch 10 Loss None
    Time taken for 1 epoch 3.174715995788574 sec
    
    [[6.67518806 9.80042553 6.56162691 ... 8.9871645 8.48175812 8.17529297]
     [8.03821 7.03849745 7.50794601 ... 7.1966629 5.9187665 3.72287822]
     [7.15120697 6.51678705 8.28324699 ... 6.1214056 9.58710289 8.34791374]
     ...
     [7.03690386 9.19953156 6.55219269 ... 7.44891262 5.52912426 7.38462925]
     [7.25152397 8.54364777 8.24735832 ... 8.74845695 5.09246349 5.81865072]
     [8.34426498 8.83569908 7.55861473 ... 4.66434765 6.53898239 6.11028767]]
    Epoch 11 Loss None
    Time taken for 1 epoch 3.2807440757751465 sec
    
    [[6.05410194 9.31308556 6.58980083 ... 6.25009871 8.76402855 7.85196543]
     [8.10052299 5.90709305 8.14972782 ... 7.40296268 5.93217087 3.07790804]
     [6.96622276 5.62584734 8.90479374 ... 6.13121462 8.94471645 8.04035]
     ...
     [6.9241333 8.63891 6.8348465 ... 7.61844397 5.78966808 7.85428476]
     [8.26302147 9.07900715 8.19392 ... 9.14385891 5.83841896 5.45720196]
     [8.5960989 8.62126541 7.35837841 ... 4.94179344 6.53101921 4.99971056]]
    Epoch 12 Loss None
    Time taken for 1 epoch 3.181723117828369 sec
    
    [[6.78932381 9.09186935 6.75122547 ... 5.13446903 7.47200632 6.5868845]
     [8.06617069 6.63512516 7.77102852 ... 7.46304893 5.71891403 3.02016211]
     [6.5288868 5.52128315 8.27158451 ... 5.89963627 8.4687624 8.18481922]
     ...
     [7.00032902 8.73995113 6.31175327 ... 8.0349865 5.94480801 7.84908438]
     [9.03041458 8.15077209 7.63351822 ... 8.06552219 4.72887373 5.8773694]
     [7.97594833 8.23747635 7.48086929 ... 4.18736 6.56665516 6.52854443]]
    Epoch 13 Loss None
    Time taken for 1 epoch 3.1927196979522705 sec
    
    [[6.80177593 9.22405243 5.8071146 ... 4.59882212 7.32495 6.13909]
     [6.6728878 7.03031635 6.62087774 ... 7.43751907 5.10649395 3.29748154]
     [6.29580259 6.13822937 8.43767929 ... 5.67493677 7.50896358 7.65373421]
     ...
     [7.14201403 9.00337887 5.98759699 ... 8.80658436 6.29934 6.67685461]
     [8.04538441 8.38694191 7.66035557 ... 8.25749 4.43737555 5.98468113]
     [7.44115353 8.30949306 7.23369122 ... 4.08551121 6.7406764 6.84069633]]
    Epoch 14 Loss None
    Time taken for 1 epoch 3.1820268630981445 sec
    
    [[6.34713173 8.67936134 5.80322361 ... 5.21220493 6.89290428 6.2013483]
     [6.27299309 6.84376097 6.37750387 ... 7.19469452 5.13483 2.90869093]
     [5.98200178 5.95033503 8.64599419 ... 4.77670479 7.41914 7.25253868]
     ...
     [6.70077467 8.52443409 5.88785 ... 7.68093395 4.97055435 6.63471222]
     [8.29554367 8.04718208 6.83317471 ... 7.82530975 4.78601027 5.26104879]
     [8.12350464 7.3521 6.67912817 ... 3.5359056 6.10372543 6.30741262]]
    Epoch 15 Loss None
    Time taken for 1 epoch 3.219971179962158 sec
    
    [[6.63236809 8.21561432 5.27575493 ... 4.77658081 6.7398572 4.79635429]
     [6.12344265 6.32821465 6.41857052 ... 7.26105213 4.80679226 2.27411318]
     [5.87663317 5.29018211 8.92214394 ... 4.32147694 6.68892956 7.02440834]
     ...
     [6.70765734 7.91226 5.89016628 ... 7.28182411 5.03198671 6.4359417]
     [7.89750576 7.76885509 6.34823227 ... 7.6462636 4.81372356 5.24974728]
     [8.13377666 7.09166 6.40835333 ... 3.4376359 5.95088768 6.07450676]]
    Epoch 16 Loss None
    Time taken for 1 epoch 3.1267197132110596 sec
    
    [[6.68654156 7.75483847 6.105896 ... 3.01311183 6.20363903 5.36239815]
     [5.92891407 6.45029449 6.64943314 ... 7.30977 5.0187006 2.01141524]
     [5.18105221 4.98072147 8.6778307 ... 4.25756359 5.18500805 6.66536427]
     ...
     [6.10899067 7.23592043 6.72547531 ... 7.220891 5.05310106 5.67614841]
     [8.3422184 7.46971464 6.1306076 ... 7.37498093 4.5445509 4.64648533]
     [8.16719055 6.63687229 6.31507635 ... 3.31431317 6.07781696 6.30821657]]
    Epoch 17 Loss None
    Time taken for 1 epoch 3.145458936691284 sec
    
    [[6.94990349 7.23079586 4.74466419 ... 2.72783017 5.04086781 3.64233065]
     [5.64099598 6.72068214 5.81548262 ... 7.63037491 4.7972517 2.78637123]
     [5.02523041 4.97282314 8.77673721 ... 5.02280521 5.30302811 6.47719336]
     ...
     [5.7414875 6.93302965 6.68001652 ... 7.26271439 4.96646118 5.71801853]
     [7.86711407 7.38784266 5.46140099 ... 6.72165 4.40627193 5.10094357]
     [7.59090805 5.42353964 6.31053 ... 3.18179464 5.95674706 6.15585089]]
    Epoch 18 Loss None
    Time taken for 1 epoch 3.175826072692871 sec
    
    [[7.01301336 6.93720818 4.427598 ... 2.2177043 4.87784386 2.96989584]
     [4.55934143 7.11774397 5.09799385 ... 7.89321899 5.0425005 2.85374045]
     [4.77647305 5.52205944 9.08912277 ... 3.89585114 4.30803967 6.20414591]
     ...
     [5.76749134 6.6829977 6.15712738 ... 7.34455299 4.30615711 5.68547106]
     [7.28316 6.65055609 5.2673378 ... 7.16358232 4.3385148 4.68294287]
     [7.50850105 4.94068575 5.90525675 ... 2.7145195 5.75065136 6.21091318]]
    Epoch 19 Loss None
    Time taken for 1 epoch 3.3311636447906494 sec
    
    [[6.80743837 6.59595299 4.93143082 ... 2.19610357 4.40790224 4.50392151]
     [4.10988092 7.2852993 4.85536098 ... 7.74016809 4.93220806 2.65090775]
     [4.40440893 5.83848619 8.45613384 ... 3.4941895 3.36917043 5.05313778]
     ...
     [5.7552228 6.10794687 6.025527 ... 6.57831955 4.37199831 5.06549263]
     [7.65159225 6.27633095 4.598979 ... 6.80512142 4.22059345 4.15697622]
     [7.58159113 4.82901764 5.70893 ... 2.39457 5.28760433 6.2789011]]
    Epoch 20 Loss None
    Time taken for 1 epoch 3.3850271701812744 sec
    
    [[6.92519045 5.49881601 5.745121 ... 1.85038018 3.6782124 4.39861107]
     [4.51896811 6.62786 4.58009529 ... 7.50498724 4.87005901 1.97212458]
     [3.91551805 5.15148687 8.14592934 ... 3.55747056 2.69510627 5.0033865]
     ...
     [5.51321125 5.52103138 6.0693469 ... 6.41657 4.00436592 4.6112318]
     [7.68149805 5.59276724 4.11372948 ... 5.89674664 4.30850506 3.90844202]
     [7.81461573 4.21345425 6.13323116 ... 3.30895591 5.14767742 5.64757347]]
    Epoch 21 Loss None
    Time taken for 1 epoch 3.502322196960449 sec
    
    [[7.14952421 5.10419273 4.98715401 ... 1.78624022 3.83664584 2.77887297]
     [4.52804565 6.58618116 4.5970993 ... 7.66212177 5.02530956 2.05524015]
     [3.5253973 4.48811913 8.14284515 ... 3.11410046 1.81183124 4.27505684]
     ...
     [5.11545086 5.09315 5.93031168 ... 6.62923622 3.73449898 4.36805725]
     [7.48282814 4.72390413 3.75060415 ... 5.78260231 4.29518223 3.8520093]
     [8.06651592 3.3055439 5.81597 ... 1.98735356 4.94208813 5.47935438]]
    Epoch 22 Loss None
    Time taken for 1 epoch 3.323240041732788 sec
    
    [[6.58386898 5.26298332 4.163095 ... 1.40154898 2.22300863 3.24937582]
     [3.97980356 6.7702322 4.17682505 ... 7.50873184 4.17185068 1.39180624]
     [3.43317199 5.13289595 8.05783844 ... 3.18232369 1.26107323 3.62151527]
     ...
     [5.31032324 4.53261042 5.56187105 ... 5.44150257 3.98754954 4.3418169]
     [6.16609287 4.87646866 4.24537182 ... 5.66329 4.24474716 3.24967098]
     [6.39116192 3.36122942 5.06913 ... 2.57550526 4.74781942 5.00206566]]
    Epoch 23 Loss None
    Time taken for 1 epoch 3.3348817825317383 sec
    
    [[6.46781921 4.10708046 4.63371944 ... 1.07034183 1.86143339 2.47141933]
     [3.26349688 6.78493452 3.57291365 ... 6.68706 3.93673587 1.82165945]
     [3.64687586 5.02843952 7.67573643 ... 3.32786417 1.52881384 3.62183547]
     ...
     [5.12193584 4.23587847 5.69925261 ... 5.70406389 3.76313329 4.22120476]
     [5.6436348 5.00464296 3.64305711 ... 4.63559628 4.00003052 3.46304154]
     [6.99437 2.37126255 4.8113184 ... 2.02438402 4.66427755 5.04567814]]
    Epoch 24 Loss None
    Time taken for 1 epoch 3.3506739139556885 sec
    
    [[6.66519547 3.09353 3.98659396 ... 1.46915317 2.18063068 2.71794796]
     [2.41091394 7.13548899 2.80614638 ... 6.66093254 3.96105504 1.28515744]
     [3.27950239 5.01213837 7.08544159 ... 2.65502644 1.0726068 3.12662601]
     ...
     [4.86893606 3.77905226 5.03162575 ... 5.2285614 3.99872875 5.02018738]
     [6.11809254 4.14643192 3.61891675 ... 4.02201796 4.12372 3.44625545]
     [5.89368296 2.10484385 5.1220541 ... 1.48717976 4.14602613 4.8635788]]
    Epoch 25 Loss None
    Time taken for 1 epoch 3.266834020614624 sec
    
    [[6.64881516 2.63610601 4.19091415 ... 1.10728657 1.92761374 1.52655]
     [2.33366466 7.35433626 2.74278688 ... 7.2275877 4.32884789 0.985416949]
     [2.9055202 5.21910095 6.69938469 ... 2.07548881 0.867894411 2.62473559]
     ...
     [4.23882389 3.10625148 5.26131678 ... 4.66097641 3.08944654 3.42685699]
     [6.84539318 3.27586126 2.77976918 ... 3.94107556 4.1412282 3.14694548]
     [6.2074213 1.83006322 4.87209606 ... 1.55000186 3.8374486 4.95303535]]
    Epoch 26 Loss None
    Time taken for 1 epoch 3.383823871612549 sec
    
    [[6.77551746 2.51176715 5.04096174 ... 0.789463341 1.50155413 1.23126125]
     [2.327034 6.65626812 2.18083715 ... 7.35657406 4.32530737 0.903676808]
     [2.48420596 5.01384735 6.60195827 ... 1.96019793 1.0453372 2.497329]
     ...
     [4.1619091 3.16184974 5.58109951 ... 4.9900732 3.16553473 2.96676922]
     [6.91047859 2.54266715 1.95330358 ... 3.09798527 4.17239475 2.6032393]
     [7.06754827 1.27892029 4.37890911 ... 1.62608218 3.74344158 6.57610321]]
    Epoch 27 Loss None
    Time taken for 1 epoch 3.4929039478302 sec
    
    [[6.94570541 2.48915696 3.0439868 ... 0.535441339 0.63809365 0.951132894]
     [2.72287703 7.12788677 2.49516487 ... 7.03157139 4.44472265 1.10993552]
     [1.87782359 4.46496964 5.47444534 ... 1.68174887 0.771015823 1.99731]
     ...
     [4.1323328 2.72208333 5.22419453 ... 5.02322 3.61253142 2.54690218]
     [6.38978767 1.72684824 1.80436349 ... 3.20806718 4.22061157 3.0552783]
     [6.6267395 1.0958401 4.0691309 ... 1.51344025 3.75024939 6.16949844]]
    Epoch 28 Loss None
    Time taken for 1 epoch 3.2893710136413574 sec
    
    [[6.61234093 2.23395276 4.20931292 ... 0.272856444 0.895302117 0.380846798]
     [1.9010725 7.33111143 1.58989143 ... 7.9453311 4.52177238 0.88376987]
     [2.07295299 4.83915424 5.15839291 ... 1.92854011 0.516180038 1.4775815]
     ...
     [3.63912702 1.97190154 4.51371527 ... 4.60851526 3.0801897 3.09120321]
     [6.22190571 1.62950969 1.64839077 ... 2.71938062 3.27837873 2.46347356]
     [5.23766518 0.568391323 3.69265604 ... 1.10344744 3.37040615 5.18483782]]
    Epoch 29 Loss None
    Time taken for 1 epoch 3.1491103172302246 sec
    
    [[6.41120434 1.5636456 3.3482604 ... 0.803577781 0.510950685 1.12088275]
     [1.14186585 6.93466282 1.41794693 ... 7.16598511 3.98645592 0.757843673]
     [2.48584366 5.08262348 5.15199 ... 1.68300486 0.265889 0.9231686]
     ...
     [3.74282503 1.38966703 4.33323097 ... 3.67722893 2.4043076 2.85493922]
     [5.02569675 1.38369501 1.43640327 ... 1.87259185 3.58727789 2.42343926]
     [5.19263411 0.628677309 3.73950768 ... 1.61850655 3.19027209 4.39013386]]
    Epoch 30 Loss None
    Time taken for 1 epoch 3.2617478370666504 sec
    
    [[6.24565697 0.922445655 3.4157486 ... 0.438225687 0.386410147 0.483726442]
     [1.41211557 6.70556927 1.4283576 ... 6.21051502 3.55413222 0.566421866]
     [1.92796957 4.63150072 4.25039577 ... 1.2484808 0.352233022 0.903546393]
     ...
     [3.43785954 1.37515545 4.54685259 ... 3.90253687 2.65812302 2.53233552]
     [4.13068104 1.27818811 1.24997938 ... 1.37928677 3.50067663 2.23093367]
     [5.57821274 0.41234076 3.05740452 ... 1.11416245 2.48674679 4.79092455]]
    Epoch 31 Loss None
    Time taken for 1 epoch 3.4547982215881348 sec
    
    [[6.29490852 1.16391659 3.4119308 ... 0.304839939 0.347578853 0.496914208]
     [0.984391212 6.66896296 1.44058061 ... 5.97779942 3.91703057 0.437649131]
     [1.47555089 4.14302158 3.74230647 ... 1.34185779 0.236508846 0.59358722]
     ...
     [3.25220394 1.18882239 4.14988232 ... 3.40748405 2.90121388 2.22188163]
     [5.13889313 0.688214779 0.904730082 ... 1.61171103 3.50047374 1.96432161]
     [4.63186836 0.249089018 2.46393919 ... 1.09226298 2.40125871 4.14038658]]
    Epoch 32 Loss None
    Time taken for 1 epoch 3.3016421794891357 sec
    
    [[6.38722 0.532500267 4.9629221 ... 0.266408145 0.678558469 0.215108991]
     [1.10394406 6.63896179 0.977120161 ... 6.6113739 3.85917521 0.388877]
     [1.18289435 4.12914467 3.35513854 ... 1.14455748 0.230206266 0.479920149]
     ...
     [2.77914858 1.17287672 4.01852846 ... 2.76519108 2.01559758 2.82613087]
     [5.0125618 0.562425494 0.732122302 ... 0.910071373 3.37443185 1.94936538]
     [4.7362361 0.297914058 1.90639448 ... 0.966581106 2.66128492 4.21762848]]
    Epoch 33 Loss None
    Time taken for 1 epoch 3.3238441944122314 sec
    
    [[6.34526634 0.868344963 2.89321017 ... 0.16622223 0.450585544 0.196879834]
     [0.917026043 6.99337912 1.02582872 ... 6.82537889 3.86014509 0.532600284]
     [0.952974558 4.81834507 4.82658291 ... 1.20631611 0.145896345 0.350789309]
     ...
     [2.61244678 1.01455283 3.81764197 ... 2.57260513 1.82634914 1.80720973]
     [6.29108238 0.5667153 0.550459504 ... 0.99616003 3.07017422 2.15333939]
     [5.28774929 0.358346909 1.86770773 ... 0.66498661 2.96057749 5.48386574]]
    Epoch 34 Loss None
    Time taken for 1 epoch 3.3541200160980225 sec
    
    [[6.6196537 0.764890432 2.58209014 ... 0.187513068 0.369360328 0.276405871]
     [0.545091271 7.35811663 0.775447 ... 6.68237686 4.01655 0.714055955]
     [1.05797303 4.69006538 3.47719765 ... 1.12632203 0.740038157 0.943118215]
     ...
     [2.67775297 0.747948229 3.82756829 ... 1.68676949 1.69956291 2.01215386]
     [5.44326496 0.801550329 0.822520614 ... 1.28055727 2.91057825 2.21572232]
     [3.75696301 0.439848334 1.95217192 ... 1.76032639 4.8741436 5.57397938]]
    Epoch 35 Loss None
    Time taken for 1 epoch 3.348797082901001 sec
    
    [[6.27341652 0.303689033 2.43647051 ... 0.21846965 0.168370023 0.387735277]
     [0.626853 6.68802404 0.443937063 ... 6.60565901 3.6348722 0.742348671]
     [1.35187054 4.42041206 2.68853498 ... 1.07036638 0.0684309229 0.261787355]
     ...
     [2.75050449 0.332614839 3.2791872 ... 1.82148731 1.84920168 1.67198169]
     [4.17893124 0.407099158 0.388655096 ... 0.665747702 2.85015655 1.57810724]
     [2.22415543 0.133462593 1.67296147 ... 0.961922 1.84590876 4.32434511]]
    Epoch 36 Loss None
    Time taken for 1 epoch 3.3498647212982178 sec
    
    [[5.56442595 1.17095232 4.34574175 ... 0.332606107 0.136792943 0.456389725]
     [1.06439328 6.37634897 0.685260773 ... 5.80427694 3.4518857 0.427175671]
     [1.37917602 4.00589323 2.65798855 ... 1.44536459 0.204529509 0.926128149]
     ...
     [2.58598661 0.417127192 3.04470205 ... 2.48092532 2.06088519 2.0958]
     [2.37776923 0.274354756 0.624377847 ... 1.10240006 3.18899465 1.4313693]
     [4.24465752 0.498040795 1.54136372 ... 1.04940164 2.48603678 4.50272]]
    Epoch 37 Loss None
    Time taken for 1 epoch 3.316824197769165 sec
    
    [[5.73995113 0.182865933 2.7306025 ... 0.116533153 0.346771806 0.129623622]
     [0.503722668 6.25339031 0.553554356 ... 5.4023037 3.34641743 0.245380491]
     [1.2219131 3.58591986 2.18776393 ... 0.537181795 0.0977625772 0.246337265]
     ...
     [1.8857013 0.483062387 3.10007572 ... 1.59335697 1.28233767 1.09609199]
     [3.63238621 0.395686954 0.482399583 ... 0.464974225 2.65204477 1.53801024]
     [2.73695493 0.126709834 1.04744029 ... 0.429334462 2.2303009 4.33410931]]
    Epoch 38 Loss None
    Time taken for 1 epoch 3.2791664600372314 sec
    
    [[6.10889053 0.250141203 2.39477563 ... 0.102494545 0.466229081 0.119442537]
     [0.364270389 6.18014431 0.491954148 ... 5.65902 3.36411715 0.250418901]
     [0.737448931 3.87532043 1.43766975 ... 0.527516842 0.0692336932 0.176534563]
     ...
     [1.5778954 0.454083383 3.15349889 ... 1.38650012 1.45221722 1.38219118]
     [3.76935053 0.285047114 0.183407307 ... 0.324003428 2.4881773 1.2454685]
     [2.44456792 0.0698446333 0.928853214 ... 0.522230506 2.32795763 3.66765428]]
    Epoch 39 Loss None
    Time taken for 1 epoch 3.296238660812378 sec
    
    [[5.79724312 0.548910797 2.60084057 ... 0.128691837 0.122903638 0.147254437]
     [0.578659892 6.41347313 0.413613945 ... 6.56165838 3.35168505 0.360753596]
     [0.600142181 3.60974693 1.93512022 ... 0.551895618 0.0818078667 0.144337058]
     ...
     [1.82513428 0.465579331 2.5539825 ... 1.02777112 1.23741877 0.750198483]
     [4.96033192 0.427016526 0.435745895 ... 0.433046073 2.26992226 1.21916]
     [2.52859139 0.129960656 0.526899219 ... 0.587239861 1.22414017 4.4932251]]
    Epoch 40 Loss None
    Time taken for 1 epoch 3.3117682933807373 sec
    
    [[5.10336304 0.17152907 2.0080471 ... 0.161956653 0.110970892 0.217447281]
     [0.346812516 6.35033751 0.318528026 ... 6.04350758 3.0927763 0.263879508]
     [0.683206141 3.2357626 1.46347296 ... 0.587619245 0.1085364 0.190174386]
     ...
     [1.34247792 0.222269028 2.12446356 ... 0.974096894 1.10803735 0.972717643]
     [2.71448898 0.0987873375 0.16642119 ... 0.665983677 2.3400743 1.23932505]
     [1.88527274 0.147799596 0.603340566 ... 0.510959685 1.55237246 4.04189253]]
    Epoch 41 Loss None
    Time taken for 1 epoch 3.4477832317352295 sec
    
    [[5.2682724 0.170443952 2.19927406 ... 0.165380329 0.150960222 0.137540922]
     [0.406500667 6.0301075 0.39668417 ... 5.18562794 3.0486455 0.235219017]
     [1.046242 3.28791475 1.50847077 ... 0.38247 0.0846643746 0.175854385]
     ...
     [1.30476582 0.18815662 2.13458395 ... 1.37111628 1.15648448 0.655662358]
     [1.83372152 0.232916489 0.255599558 ... 0.180228278 1.9294486 1.14289892]
     [0.888663769 0.0622128546 0.37429148 ... 0.505771637 1.34618878 3.69453669]]
    Epoch 42 Loss None
    Time taken for 1 epoch 3.2477381229400635 sec
    
    [[5.44036198 0.138249204 2.11398315 ... 0.0760230795 0.120573618 0.0739450678]
     [0.377502352 5.89643097 0.244197309 ... 5.08311749 2.69677734 0.204953432]
     [0.599216 2.90087271 1.04669619 ... 0.608174384 0.0724164 0.096918121]
     ...
     [1.15603316 0.205631182 1.41234076 ... 0.448095083 1.18725502 0.738422513]
     [1.52358925 0.0933371857 0.156018376 ... 0.189594671 1.94306958 0.695818424]
     [1.10582078 0.0562051907 0.526057 ... 0.483828306 1.10687876 3.1964879]]
    Epoch 43 Loss None
    Time taken for 1 epoch 3.264759063720703 sec
    
    [[5.55402613 0.194517046 2.16066909 ... 0.0916709527 0.124136426 0.0732847825]
     [0.268238634 5.90170383 0.398026258 ... 5.35105324 2.89471912 0.217536181]
     [0.631963968 2.81292582 1.20575428 ... 0.284390897 0.0712125 0.0952201411]
     ...
     [1.02766871 0.268958658 1.85673404 ... 0.946219921 1.17719007 0.744287252]
     [1.59760869 0.105244361 0.165327489 ... 0.219582975 1.75169945 1.039011]
     [0.847587228 0.0805368051 0.306999773 ... 0.406927 1.19449675 3.9731679]]
    Epoch 44 Loss None
    Time taken for 1 epoch 3.2477328777313232 sec
    
    [[4.84233379 0.169183046 1.96299672 ... 0.0940382928 0.0990708917 0.0985827744]
     [0.323756307 5.46346951 0.431333095 ... 5.17278481 2.94574428 0.215266153]
     [0.520604372 2.75288844 0.964773774 ... 0.234564334 0.0642612576 0.0924895257]
     ...
     [0.772243559 0.154045597 1.45408046 ... 0.455123127 0.727547765 0.407955945]
     [2.10976267 0.077237986 0.0864551514 ... 0.361783266 1.5479027 0.852896571]
     [0.687054396 0.065281719 0.266893536 ... 0.323829323 0.947703421 3.63858938]]
    Epoch 45 Loss None
    Time taken for 1 epoch 3.376859188079834 sec
    
    [[4.78459167 0.16196081 1.34503579 ... 0.0659254268 0.0705246329 0.0858089849]
     [0.398240328 5.60880947 0.261870593 ... 5.25459862 2.65675831 0.209280282]
     [0.458155096 2.82691836 0.737651348 ... 0.35834083 0.0767776147 0.118376575]
     ...
     [0.635574341 0.141103938 1.30645144 ... 1.08259702 1.2204833 0.642948329]
     [1.37646937 0.0717724115 0.115415864 ... 0.125833362 1.40041494 1.46935201]
     [0.528012633 0.0356749147 0.245852143 ... 0.627599955 1.04124415 2.87677336]]
    Epoch 46 Loss None
    Time taken for 1 epoch 3.283740282058716 sec
    
    [[4.59522486 0.0951434 1.68604708 ... 0.0573159643 0.087556 0.0464692041]
     [0.191151172 5.44931173 0.221937478 ... 4.62185383 2.5388422 0.139798507]
     [0.490958393 2.16573763 0.593196809 ... 0.261766344 0.0580367073 0.0986784697]
     ...
     [0.614977181 0.104689248 0.785741448 ... 0.33570388 1.19644177 0.404406041]
     [0.851280808 0.0898279399 0.155958608 ... 0.128168687 1.17229 0.741761088]
     [0.542498529 0.0732767 0.347922802 ... 0.325870097 0.742868602 2.63686466]]
    Epoch 47 Loss None
    Time taken for 1 epoch 3.3292737007141113 sec
    
    [[4.85271406 0.109891906 1.93015492 ... 0.0639397055 0.129282922 0.0624606088]
     [0.201581046 5.22362852 0.208956867 ... 4.34093618 2.55806446 0.153167]
     [0.488683939 2.16053724 0.755932391 ... 0.414329052 0.0799268633 0.0833422169]
     ...
     [0.473663211 0.147450209 0.753214717 ... 0.695695817 0.850530386 0.721262157]
     [0.712315679 0.0650979951 0.0770752057 ... 0.131360516 1.20745349 0.663405299]
     [0.327263355 0.0547558 0.193953484 ... 0.382710934 0.936785698 3.44685149]]
    Epoch 48 Loss None
    Time taken for 1 epoch 3.2437407970428467 sec
    
    [[4.37594604 0.118241116 1.27452481 ... 0.054361511 0.0673857 0.0541236]
     [0.250582516 5.14399719 0.314644516 ... 4.43080235 2.44163752 0.169219077]
     [0.306739628 2.22660398 0.732794523 ... 0.218205363 0.0401499607 0.0746576935]
     ...
     [0.400088638 0.120499648 0.940981507 ... 0.235708281 0.748605072 0.295197666]
     [1.25027668 0.0886151567 0.0731464 ... 0.140581027 1.03917348 0.626785457]
     [0.553896308 0.040441826 0.166608304 ... 0.33502534 0.666025 2.51628256]]
    Epoch 49 Loss None
    Time taken for 1 epoch 3.329977035522461 sec
    
    [[4.37313175 0.0937814191 1.03794205 ... 0.040104039 0.0983269513 0.0443258286]
     [0.290803164 4.78920269 0.277243555 ... 4.61378145 2.32940245 0.169137746]
     [0.292337924 2.19577742 0.313405782 ... 0.177211568 0.0385448523 0.0760590956]
     ...
     [0.415683627 0.126096636 0.640468538 ... 0.338115185 0.741728842 0.320644706]
     [0.347892076 0.118606456 0.125532687 ... 0.141255483 0.947538555 0.698030233]
     [0.296425581 0.028734386 0.219323665 ... 0.428818583 0.832155704 2.15935636]]
    Epoch 50 Loss None
    Time taken for 1 epoch 3.387333631515503 sec
    
    [[4.33894825 0.103226468 1.15854 ... 0.0505783595 0.0862632468 0.0518204793]
     [0.177414149 4.69633627 0.205882803 ... 4.19570637 2.14023495 0.200187773]
     [0.378974646 1.7773124 0.433857501 ... 0.24581261 0.0397165343 0.0676753]
     ...
     [0.363730282 0.0912886709 0.468643367 ... 0.309270293 0.698727131 0.196542501]
     [0.444518954 0.0470333621 0.111082479 ... 0.106217645 0.603073 0.581751108]
     [0.227719665 0.0472245254 0.230274633 ... 0.261111557 0.583768964 2.2725184]]
    Epoch 51 Loss None
    Time taken for 1 epoch 3.4539682865142822 sec
    
    [[3.99035287 0.094068028 1.606251 ... 0.0489481427 0.067320846 0.0472819507]
     [0.146524712 4.50032043 0.160761625 ... 3.79857469 2.07075787 0.140642747]
     [0.281415462 1.44162834 0.604898512 ... 0.271674 0.0440365784 0.0569118708]
     ...
     [0.330354869 0.119735315 0.337012261 ... 0.23946856 0.429013908 0.249883488]
     [0.471629679 0.0444602743 0.0813450888 ... 0.121059574 0.777830064 0.580205679]
     [0.320894659 0.0244276188 0.117582217 ... 0.263333291 0.705089271 2.26625586]]
    Epoch 52 Loss None
    Time taken for 1 epoch 3.2347304821014404 sec
    
    [[3.57494092 0.0569573678 0.990146637 ... 0.0408877581 0.0668780729 0.035957735]
     [0.161277741 4.44473743 0.134479567 ... 3.53637958 2.20437956 0.0988929495]
     [0.265001059 1.48960817 0.339264721 ... 0.165599555 0.0406153686 0.0823420137]
     ...
     [0.301516384 0.0870306417 0.407190412 ... 0.209579229 0.499032289 0.235209122]
     [0.515941679 0.0565060414 0.0660237521 ... 0.0925372392 0.751466 0.446145356]
     [0.335674137 0.0437999628 0.22161679 ... 0.194958955 0.543327391 1.96410108]]
    Epoch 53 Loss None
    Time taken for 1 epoch 3.4044148921966553 sec
    
    [[3.95514512 0.063644439 0.940315247 ... 0.0340384021 0.0631378144 0.0407653078]
     [0.15663144 4.09249401 0.15777114 ... 3.47520924 1.83215165 0.0989440233]
     [0.244915143 1.41613388 0.301101327 ... 0.196288124 0.0333097167 0.0646994933]
     ...
     [0.313818365 0.0807696208 0.400246352 ... 0.189412892 0.549997687 0.191520959]
     [0.544779778 0.0397750735 0.0577105507 ... 0.102888055 0.668407261 0.431356549]
     [0.137975395 0.0253256522 0.147117376 ... 0.303433478 0.685461581 1.93109822]]
    Epoch 54 Loss None
    Time taken for 1 epoch 3.276740312576294 sec
    
    [[3.96350694 0.0582553595 0.878260851 ... 0.0395578444 0.0645091832 0.0418936945]
     [0.133913338 3.97785425 0.151402846 ... 3.62206268 1.91899753 0.1527448]
     [0.262051851 1.16007161 0.304491341 ... 0.185643 0.0344425105 0.0624509789]
     ...
     [0.273320407 0.0798631459 0.302742362 ... 0.1510043 0.48274231 0.304284781]
     [0.160952136 0.0431670733 0.0754536 ... 0.0961314291 0.457851589 0.491738617]
     [0.191306159 0.0315981805 0.131630599 ... 0.191841096 0.453730941 1.56758523]]
    Epoch 55 Loss None
    Time taken for 1 epoch 3.2437329292297363 sec
    
    [[3.34160781 0.0601461 0.865746379 ... 0.0386394784 0.0604190566 0.0335973576]
     [0.120795704 3.72592664 0.120039426 ... 3.10953236 1.66436839 0.111098588]
     [0.255532533 1.06955075 0.3322348 ... 0.159643725 0.0334798917 0.0507290773]
     ...
     [0.256504923 0.0591602772 0.232824028 ... 0.118903719 0.300978065 0.208912745]
     [0.332817316 0.0329509415 0.0628161207 ... 0.102724448 0.510985732 0.510564446]
     [0.152026847 0.0204290785 0.111677982 ... 0.191959277 0.491808653 1.40741813]]
    Epoch 56 Loss None
    Time taken for 1 epoch 3.2500624656677246 sec
    
    [[3.15137267 0.0429992266 0.70802778 ... 0.0328487419 0.0552915037 0.033090502]
     [0.141368613 3.41683769 0.139292851 ... 2.71078825 1.64074647 0.115597904]
     [0.262502313 0.968502283 0.267857671 ... 0.143037915 0.0327743329 0.0627775]
     ...
     [0.245335162 0.051352784 0.25135687 ... 0.198083162 0.253642082 0.166854605]
     [0.30074 0.0410180874 0.0546027534 ... 0.079694517 0.546993613 0.305296391]
     [0.396811485 0.0476240255 0.194084063 ... 0.240522578 0.504993081 1.28157938]]
    Epoch 57 Loss None
    Time taken for 1 epoch 3.310530185699463 sec
    
    [[3.31724024 0.0482551232 0.657059968 ... 0.0392810814 0.0510610305 0.0359924659]
     [0.136333808 3.34914112 0.0999823511 ... 2.59311604 1.43824911 0.0784457922]
     [0.257554054 0.895388782 0.354616642 ... 0.119796015 0.0340555683 0.0631406158]
     ...
     [0.266442209 0.0661460459 0.248790041 ... 0.202427894 0.170291841 0.145019919]
     [0.227196828 0.0356383324 0.0609969348 ... 0.0685838684 0.47467947 0.233684361]
     [0.0646647438 0.0212885812 0.0957319 ... 0.230526373 0.400687158 1.18346214]]
    Epoch 58 Loss None
    Time taken for 1 epoch 3.2717394828796387 sec
    
    [[3.12706041 0.0444646068 0.764878094 ... 0.032719072 0.0438379571 0.0379979201]
     [0.11798276 3.0643816 0.105172791 ... 2.55761242 1.53563094 0.0931852609]
     [0.24292326 0.936760068 0.280225724 ... 0.133483142 0.0330801196 0.0593604855]
     ...
     [0.235992938 0.0569648 0.240180805 ... 0.16655381 0.221453458 0.177833542]
     [0.170755446 0.0302367639 0.0596574657 ... 0.0673282 0.424366236 0.288972944]
     [0.165176809 0.0275854766 0.137432456 ... 0.154877082 0.31150502 1.02423179]]
    Epoch 59 Loss None
    Time taken for 1 epoch 3.55700421333313 sec
    
    [[2.83503366 0.0363903828 0.607081711 ... 0.0365746282 0.0584725 0.0328561254]
     [0.118259549 2.93672967 0.122158021 ... 2.39750791 1.36489022 0.129282713]
     [0.22252214 0.807412624 0.306294978 ... 0.112842366 0.0265476257 0.046837952]
     ...
     [0.214756 0.0374869704 0.198434055 ... 0.0958794281 0.213437498 0.21712774]
     [0.219972804 0.0335592031 0.045786988 ... 0.0533782169 0.31735006 0.234246343]
     [0.103296675 0.0161541179 0.10002733 ... 0.169615358 0.354555845 0.902075291]]
    Epoch 60 Loss None
    Time taken for 1 epoch 3.273740768432617 sec
    
    [[2.69570374 0.037838269 0.774231672 ... 0.0297932308 0.0432513319 0.0310595687]
     [0.106197171 2.6114037 0.151101053 ... 2.05690551 1.21012139 0.110755786]
     [0.313417196 0.619359314 0.282999903 ... 0.0969437659 0.02687563 0.0430386215]
     ...
     [0.210377976 0.0414832048 0.196702227 ... 0.159754992 0.156943291 0.115979493]
     [0.214593947 0.0344451629 0.0528999455 ... 0.0652456433 0.349118352 0.228517354]
     [0.122347377 0.0278304834 0.125393361 ... 0.193622619 0.333719283 0.827232957]]
    Epoch 61 Loss None
    Time taken for 1 epoch 3.5321104526519775 sec
    
    [[2.41819024 0.0408316851 0.498059452 ... 0.0288994331 0.0333859287 0.0308945309]
     [0.0902615339 2.48695374 0.0743903667 ... 1.98451793 1.15207756 0.104871534]
     [0.260969043 0.589605629 0.228968188 ... 0.120840393 0.0223348178 0.03757561]
     ...
     [0.209309086 0.0466458 0.186127648 ... 0.141202793 0.13917233 0.117737256]
     [0.144718364 0.0277027115 0.0425553769 ... 0.0615133867 0.278939873 0.216380447]
     [0.0815376118 0.0172938481 0.086077325 ... 0.209423184 0.356059432 0.875639439]]
    Epoch 62 Loss None
    Time taken for 1 epoch 3.42220401763916 sec
    
    [[2.15965343 0.0347905047 0.394566 ... 0.0258484259 0.0314080492 0.0302248523]
     [0.0876881406 2.1977582 0.0971255153 ... 1.69410729 1.1909132 0.0930766538]
     [0.190557435 0.573886573 0.180659339 ... 0.127886966 0.0214002561 0.0368432961]
     ...
     [0.165812269 0.0392088704 0.185007349 ... 0.101027578 0.191222072 0.114073791]
     [0.166342348 0.0310742464 0.0389716811 ... 0.05040529 0.282233924 0.260975391]
     [0.0785986483 0.0193041433 0.086559996 ... 0.141344383 0.276525497 0.767585337]]
    Epoch 63 Loss None
    Time taken for 1 epoch 3.2427361011505127 sec
    
    [[1.91587198 0.0452084541 0.421644211 ... 0.0237162244 0.0305925831 0.0314369276]
     [0.0806127 2.17023611 0.103584439 ... 1.55997074 1.06087637 0.127939209]
     [0.138931468 0.510906816 0.122248851 ... 0.104465485 0.0200298987 0.0323081464]
     ...
     [0.135592833 0.0376789384 0.172226846 ... 0.0890729427 0.183453351 0.112589955]
     [0.118797973 0.029815793 0.0333033763 ... 0.0550833717 0.247790202 0.277431369]
     [0.0685240924 0.0151069099 0.0537941307 ... 0.163929269 0.284053028 0.785457492]]
    Epoch 64 Loss None
    Time taken for 1 epoch 3.275815010070801 sec
    
    [[2.09537864 0.0393595919 0.456130445 ... 0.0218900964 0.0319082551 0.0281636324]
     [0.0707613453 1.87107658 0.0909387916 ... 1.30332339 0.826397538 0.0980438292]
     [0.1245974 0.363720268 0.135789037 ... 0.0925798342 0.0216402486 0.031633988]
     ...
     [0.121129595 0.0448329523 0.146925762 ... 0.0876392126 0.118418507 0.0965683684]
     [0.154546306 0.0231607687 0.035819497 ... 0.0529217683 0.274590105 0.264670104]
     [0.0857562497 0.0170272626 0.0801702738 ... 0.144204244 0.388259 0.801885903]]
    Epoch 65 Loss None
    Time taken for 1 epoch 3.347158908843994 sec
    
    [[1.59557307 0.0435322449 0.419729054 ... 0.0209537186 0.0319331959 0.0241474509]
     [0.0665473193 1.75237381 0.0755486712 ... 1.42662442 0.780182958 0.0790242553]
     [0.0810426101 0.356772184 0.128730938 ... 0.116600521 0.0178200174 0.0297722872]
     ...
     [0.109708294 0.0368074477 0.140858456 ... 0.0797483474 0.210347742 0.095093]
     [0.105955511 0.0278379023 0.0355147757 ... 0.0521237776 0.284456283 0.262170464]
     [0.0912674591 0.0153675526 0.0734980181 ... 0.15529798 0.248070046 0.579642355]]
    Epoch 66 Loss None
    Time taken for 1 epoch 3.2886295318603516 sec
    
    [[1.32739866 0.0256869402 0.323745161 ... 0.0186002087 0.0319596343 0.0213343259]
     [0.0626615 1.61846805 0.073839888 ... 1.10289633 0.675966263 0.0644319504]
     [0.0830762163 0.367827088 0.130816892 ... 0.0947118104 0.0151699632 0.0313841328]
     ...
     [0.10382054 0.0364657864 0.166583672 ... 0.0764613822 0.115124263 0.0988880917]
     [0.130687192 0.023588974 0.0325030722 ... 0.044238355 0.239786476 0.235511154]
     [0.061137341 0.017275922 0.0694417953 ... 0.116459414 0.327041179 0.533105731]]
    Epoch 67 Loss None
    Time taken for 1 epoch 3.3033552169799805 sec
    
    [[1.75005937 0.0199625939 0.296737164 ... 0.0209506825 0.028001707 0.0221531764]
     [0.0575690977 1.49649501 0.0586029217 ... 1.02398074 0.661482036 0.0797570422]
     [0.138878033 0.299147218 0.106070563 ... 0.090538919 0.0180614553 0.0337965265]
     ...
     [0.0938588083 0.0381503142 0.121272795 ... 0.0536633097 0.0893166363 0.117114604]
     [0.0948350951 0.0228469223 0.0403999239 ... 0.0409691148 0.179997206 0.201794341]
     [0.0507550277 0.0167336408 0.0717071742 ... 0.138602957 0.229313254 0.388807267]]
    Epoch 68 Loss None
    Time taken for 1 epoch 3.2237277030944824 sec
    
    [[2.11719561 0.0212109741 0.389278501 ... 0.0274344794 0.034547545 0.0217650644]
     [0.0581152178 1.3351053 0.0700274855 ... 0.811878264 0.547248363 0.0769577697]
     [0.155683488 0.198465139 0.139746055 ... 0.0783044845 0.0194090232 0.0294297189]
     ...
     [0.121160746 0.0394353457 0.112439439 ... 0.0648853 0.0944948047 0.107619651]
     [0.0854895785 0.0217488501 0.0385667607 ... 0.0439164527 0.159437 0.15176098]
     [0.0691648349 0.0162039679 0.0795391873 ... 0.130000845 0.244308144 0.381956965]]
    Epoch 69 Loss None
    Time taken for 1 epoch 3.310166597366333 sec
    
    [[1.73565257 0.0190000776 0.330680907 ... 0.0320651606 0.0249894634 0.0219628699]
     [0.063149564 1.21723056 0.0727985352 ... 0.965595961 0.572754323 0.094559662]
     [0.20774439 0.231331661 0.174237117 ... 0.0713621378 0.015807908 0.026928084]
     ...
     [0.143058896 0.0299075451 0.0988299921 ... 0.0860241652 0.081313 0.0853900909]
     [0.0741054788 0.0183262452 0.0318251178 ... 0.0526858866 0.182298154 0.124493517]
     [0.0479720086 0.0148560647 0.0728204846 ... 0.140731603 0.191125676 0.436643898]]
    Epoch 70 Loss None
    Time taken for 1 epoch 3.290743112564087 sec
    
    [[1.45955992 0.0246665366 0.300079346 ... 0.0226890296 0.0281230658 0.0235831533]
     [0.0744548813 1.01429963 0.0597677417 ... 0.892137766 0.618433952 0.0536321253]
     [0.169338644 0.249554098 0.133569404 ... 0.0608138815 0.0152078876 0.0234371386]
     ...
     [0.11820405 0.0234948955 0.0976446271 ... 0.0911629945 0.0686805919 0.0782693177]
     [0.065720059 0.0193127953 0.0311427712 ... 0.0394866839 0.167957023 0.0912657157]
     [0.0547078364 0.0138074961 0.0684473962 ... 0.145495489 0.183876067 0.380703777]]
    Epoch 71 Loss None
    Time taken for 1 epoch 3.4197728633880615 sec
    
    [[1.16910303 0.0397213437 0.282322556 ... 0.017049998 0.0245579034 0.0244856663]
     [0.0705014169 1.04269361 0.0556880198 ... 0.920636535 0.489618272 0.0644925237]
     [0.0863464624 0.247220695 0.105772898 ... 0.0802859 0.0146742333 0.0207762606]
     ...
     [0.0898676 0.0289745945 0.108582065 ... 0.0784874484 0.142082676 0.0833235681]
     [0.0474357791 0.0211268254 0.0255484357 ... 0.0338271819 0.164058477 0.152406976]
     [0.0598062575 0.013066384 0.0416550823 ... 0.137256324 0.186780989 0.405614108]]
    Epoch 72 Loss None
    Time taken for 1 epoch 3.331141233444214 sec
    
    [[0.853243589 0.0247493424 0.270622432 ... 0.0144312708 0.0245149806 0.0204776656]
     [0.0512967259 1.07821059 0.0714079812 ... 0.819835305 0.408726126 0.0855286494]
     [0.0584953241 0.209483907 0.0981965289 ... 0.0648038536 0.0130040217 0.022077268]
     ...
     [0.069268845 0.0399030298 0.103621945 ... 0.0556379557 0.150552273 0.080592677]
     [0.0692174509 0.0186178777 0.0239156242 ... 0.032212574 0.154093936 0.13400884]
     [0.108370401 0.0158284418 0.0500682481 ... 0.0999510735 0.197199941 0.407904506]]
    Epoch 73 Loss None
    Time taken for 1 epoch 3.341761827468872 sec
    
    [[0.938548267 0.0142564131 0.198669508 ... 0.015185697 0.0225564055 0.0189542267]
     [0.0359383 0.817834854 0.049268432 ... 0.737630725 0.359012 0.0737891719]
     [0.082834281 0.161338508 0.0789980367 ... 0.0638251826 0.0134722348 0.0229273159]
     ...
     [0.0671231151 0.0351153575 0.103661604 ... 0.0378036052 0.0857229903 0.0875174403]
     [0.132019281 0.0207328256 0.0294144396 ... 0.031672217 0.140074924 0.112219]
     [0.0539073162 0.0158814769 0.0561996698 ... 0.0965850353 0.191066489 0.318646163]]
    Epoch 74 Loss None
    Time taken for 1 epoch 3.3356854915618896 sec
    
    [[1.34041691 0.0132009806 0.294527054 ... 0.021529302 0.0240964815 0.0204037335]
     [0.0417396948 0.690511882 0.05471991 ... 0.614245713 0.256956607 0.0669512153]
     [0.106712982 0.16603969 0.0874343216 ... 0.0626597106 0.0152060091 0.0238894355]
     ...
     [0.0689453259 0.0268180687 0.0755368397 ... 0.0459883 0.0872123465 0.0937467963]
     [0.0841787 0.0151692592 0.02723868 ... 0.0337666757 0.126327798 0.14445582]
     [0.0509162545 0.0118393609 0.0596234351 ... 0.123040043 0.207372919 0.284997642]]
    Epoch 75 Loss None
    Time taken for 1 epoch 3.2357311248779297 sec
    
    [[1.0615989 0.0157762263 0.239666879 ... 0.0224649087 0.0190736465 0.019287305]
     [0.0528603718 0.598703861 0.0618301257 ... 0.69345814 0.386699915 0.0697563589]
     [0.117498167 0.154500753 0.0848952383 ... 0.0575880036 0.0109591959 0.0193485748]
     ...
     [0.0906275511 0.0195318963 0.0745309 ... 0.0612547472 0.0588945076 0.0473762043]
     [0.0463727 0.0153460698 0.0269627795 ... 0.0393318571 0.125579685 0.101317175]
     [0.0436841436 0.0138090253 0.058212623 ... 0.108291782 0.169373155 0.248434514]]
    Epoch 76 Loss None
    Time taken for 1 epoch 3.2367310523986816 sec
    
    [[0.684191108 0.0244466979 0.187979519 ... 0.015802158 0.0227616299 0.0198163986]
     [0.0526680164 0.648978174 0.0445921905 ... 0.624980211 0.401988894 0.0536927953]
     [0.066527687 0.136842951 0.0793567449 ... 0.057925 0.0107041243 0.0172766242]
     ...
     [0.0758629814 0.0170621853 0.0763026774 ... 0.0589948669 0.0596238859 0.0549149178]
     [0.04013725 0.0137255434 0.0196982361 ... 0.0279927831 0.120914549 0.0913267583]
     [0.0272237156 0.0114841117 0.0446787216 ... 0.112697743 0.132962897 0.270686269]]
    Epoch 77 Loss None
    Time taken for 1 epoch 3.322999954223633 sec
    
    [[0.57437706 0.0212225281 0.20015049 ... 0.0128279729 0.0240509789 0.0178984739]
     [0.0402336717 0.673994541 0.0437568314 ... 0.5961169 0.29203558 0.0625618473]
     [0.0462703817 0.141739875 0.0737788752 ... 0.0456944034 0.0111633958 0.0172030404]
     ...
     [0.0591251068 0.0260246359 0.0813578367 ... 0.0448954143 0.0904095471 0.0705484077]
     [0.0517855026 0.0162378643 0.0199282374 ... 0.0270196367 0.116787538 0.111727662]
     [0.0563658811 0.0122493599 0.0409283824 ... 0.0804764256 0.118625626 0.243069008]]
    Epoch 78 Loss None
    Time taken for 1 epoch 3.313138961791992 sec
    
    [[0.708526 0.0105086789 0.173374176 ... 0.0143110594 0.0170107372 0.0172005799]
     [0.0307261124 0.524841547 0.0371433608 ... 0.471461177 0.199952871 0.0583880544]
     [0.0636856 0.119626164 0.0630630553 ... 0.0480142757 0.011317932 0.0173521955]
     ...
     [0.0495729595 0.0263933353 0.0730121 ... 0.0270721968 0.0563800782 0.0666248277]
     [0.0959096476 0.0145052932 0.0225348435 ... 0.0231540129 0.099315621 0.0901708]
     [0.052811522 0.0111618629 0.0473345928 ... 0.0745918602 0.141770393 0.214229435]]
    Epoch 79 Loss None
    Time taken for 1 epoch 3.302961826324463 sec
    
    [[0.879667044 0.0103923604 0.182605669 ... 0.0186502896 0.0162948668 0.0177647434]
     [0.0361881666 0.402279228 0.0419150703 ... 0.444113493 0.236192241 0.054997623]
     [0.0861315727 0.133083224 0.0669980347 ... 0.0434467569 0.0116740651 0.0163807124]
     ...
     [0.0595387518 0.0225071032 0.0573266558 ... 0.0426790938 0.0631326661 0.0463398099]
     [0.0339442641 0.0137930345 0.0208001956 ... 0.0260248687 0.0960993767 0.0840856582]
     [0.0415621139 0.0117097655 0.0522058122 ... 0.0836868733 0.15178588 0.184590548]]
    Epoch 80 Loss None
    Time taken for 1 epoch 3.22672963142395 sec
    
    [[0.470469564 0.0150678074 0.14161934 ... 0.0143405553 0.0176033527 0.0167773664]
     [0.0442326516 0.401936889 0.0373341292 ... 0.567120314 0.250554472 0.0420951024]
     [0.0627291277 0.110564969 0.0638775229 ... 0.0446079262 0.00886740908 0.0139106037]
     ...
     [0.0603783205 0.015269056 0.0567548759 ... 0.043109186 0.0411085896 0.0380388908]
     [0.0330909602 0.010224347 0.0183567945 ... 0.0232398566 0.0881122947 0.0750658438]
     [0.0282471906 0.0101877674 0.0420030914 ... 0.0786506608 0.115842231 0.190469354]]
    Epoch 81 Loss None
    Time taken for 1 epoch 3.3947668075561523 sec
    
    [[0.455504656 0.0130850924 0.123480029 ... 0.0127986679 0.016691437 0.0158573072]
     [0.0349025242 0.37019223 0.0320050083 ... 0.381006598 0.215221748 0.0545231737]
     [0.0481935553 0.102241233 0.0546182171 ... 0.0344588682 0.00934264157 0.0147011345]
     ...
     [0.0507544614 0.0161988065 0.0559506975 ... 0.0373319499 0.0478033721 0.0422495045]
     [0.0491078459 0.011102329 0.0172649082 ... 0.0213736501 0.0861903057 0.0776271075]
     [0.0347434133 0.00943499245 0.0385484099 ... 0.0655439 0.108850673 0.181824073]]
    Epoch 82 Loss None
    Time taken for 1 epoch 3.2647430896759033 sec
    
    [[0.464711666 0.00961836893 0.128703788 ... 0.0128919939 0.0178581923 0.0158385336]
     [0.0303461719 0.318933219 0.0313919894 ... 0.344250113 0.170461252 0.0504470021]
     [0.0516202264 0.0976685211 0.0509928353 ... 0.0360375419 0.00967385899 0.0140598975]
     ...
     [0.0478131473 0.017182067 0.0535054617 ... 0.0245584846 0.0454516858 0.052866023]
     [0.05565577 0.0131495679 0.0203295611 ... 0.0191296674 0.0789609104 0.0679498836]
     [0.0344536826 0.00947431475 0.0416893847 ... 0.0645749 0.114422902 0.166324392]]
    Epoch 83 Loss None
    Time taken for 1 epoch 3.2907421588897705 sec
    
    [[0.425257832 0.0100374231 0.109731473 ... 0.014042736 0.0140836416 0.0153466566]
     [0.0359891281 0.281761229 0.0325964242 ... 0.349018455 0.176521569 0.0490985401]
     [0.0534312204 0.0942873 0.052833572 ... 0.0352085829 0.00879970286 0.0132919168]
     ...
     [0.049619358 0.015582121 0.0482412651 ... 0.0380392335 0.0401197299 0.0343725979]
     [0.0345678143 0.00954988692 0.0170946475 ... 0.0188964419 0.0742499232 0.0651156381]
     [0.0283950549 0.00933106802 0.0414892659 ... 0.0644367561 0.104438856 0.156207755]]
    Epoch 84 Loss None
    Time taken for 1 epoch 3.316028118133545 sec
    
    [[0.364951313 0.0102090072 0.113474287 ... 0.0127859572 0.0176780783 0.0150956372]
     [0.0332061686 0.27483812 0.0297063291 ... 0.33618328 0.170645595 0.0432721078]
     [0.0462307706 0.0894574896 0.0485822186 ... 0.0332970321 0.00787581783 0.0125852674]
     ...
     [0.0464931019 0.0144833224 0.0480407476 ... 0.0345211737 0.038386438 0.0342175551]
     [0.0354157127 0.00921508484 0.0158032142 ... 0.0184563901 0.0726043284 0.0649579167]
     [0.029164033 0.00852611382 0.0377665274 ... 0.0603020042 0.0961958617 0.152441874]]
    Epoch 85 Loss None
    Time taken for 1 epoch 3.2567505836486816 sec
    
    [[0.364395678 0.0097387908 0.101453759 ... 0.0125470078 0.0134008406 0.0138778035]
     [0.0307332799 0.244140625 0.0295558814 ... 0.307275981 0.161431745 0.0490396321]
     [0.0446930863 0.0851259455 0.0467164516 ... 0.0322565548 0.00790302083 0.0123697016]
     ...
     [0.0437364057 0.0149109131 0.0466702618 ... 0.0323464647 0.0383514464 0.0357318521]
     [0.0435125 0.00929658115 0.0159469452 ... 0.0173527803 0.0707113594 0.0618718117]
     [0.0274583716 0.00825292338 0.0375281908 ... 0.0578447841 0.0933178589 0.144535154]]
    Epoch 86 Loss None
    Time taken for 1 epoch 3.211726188659668 sec
    
    [[0.345747322 0.00932728872 0.0957598463 ... 0.012375 0.0132187447 0.0135433888]
     [0.0305161569 0.234205708 0.0288334172 ... 0.291221172 0.152443409 0.0466373824]
     [0.0428938195 0.0812436491 0.0446843058 ... 0.031196041 0.00771483406 0.012011461]
     ...
     [0.0426817201 0.0140977465 0.044679977 ... 0.031108452 0.0370592773 0.0337303728]
     [0.0362754278 0.0085663 0.0152157536 ... 0.0166324656 0.0670943558 0.0585874058]
     [0.0295656025 0.00788220484 0.0362430066 ... 0.0569614209 0.0900792927 0.135975]]
    Epoch 87 Loss None
    Time taken for 1 epoch 3.206723690032959 sec
    
    [[0.315294236 0.00873163808 0.0971576497 ... 0.0120295994 0.0156778861 0.013673217]
     [0.0302401185 0.213096172 0.0285291094 ... 0.292872727 0.148315862 0.0415566266]
     [0.040537186 0.0785758346 0.0431066751 ... 0.0301052518 0.0072613135 0.0113406805]
     ...
     [0.0416641161 0.0138239572 0.0432598926 ... 0.0294945389 0.0360092521 0.0326039232]
     [0.0336987935 0.00978943519 0.0163501054 ... 0.0162037332 0.0651057 0.0566920228]
     [0.0282252878 0.00782046374 0.0327566825 ... 0.0538721867 0.0833278447 0.131012455]]
    Epoch 88 Loss None
    Time taken for 1 epoch 3.201724052429199 sec
    
    [[0.305077732 0.00886268262 0.0899373367 ... 0.0118453689 0.0122778574 0.0126650771]
     [0.0287223402 0.207371175 0.0274137165 ... 0.258809417 0.144217968 0.0445305109]
     [0.0395550951 0.0754026398 0.0422659591 ... 0.0300562046 0.00688762264 0.0111070443]
     ...
     [0.0396123864 0.0132299205 0.041787602 ... 0.0287028793 0.0366063491 0.0313709639]
     [0.0352904052 0.00830860715 0.0148198893 ... 0.0155837638 0.0630492866 0.0556663722]
     [0.0275423378 0.0073507796 0.0332051292 ... 0.0524669178 0.0812380463 0.124394886]]
    Epoch 89 Loss None
    Time taken for 1 epoch 3.2027225494384766 sec
    
    [[0.290775716 0.00819640793 0.0919228271 ... 0.0115602389 0.0119635211 0.0123490961]
     [0.0272396076 0.193628311 0.0265872106 ... 0.245464519 0.132359058 0.0395749174]
     [0.0365702622 0.0727191716 0.0409926884 ... 0.0280964077 0.00679290527 0.0106920945]
     ...
     [0.0384271629 0.0128013752 0.0412235633 ... 0.0270987675 0.0343362 0.0313208252]
     [0.0346553288 0.007777764 0.0136783915 ... 0.0149608264 0.0608132072 0.0530810617]
     [0.0262443628 0.00710259797 0.0316998176 ... 0.0496107377 0.0788300335 0.120512746]]
    Epoch 90 Loss None
    Time taken for 1 epoch 3.2077255249023438 sec
    
    [[0.265561163 0.00835175812 0.0808850676 ... 0.0112998979 0.0117171882 0.0119913192]
     [0.0275390893 0.181496039 0.0259746909 ... 0.243814662 0.131210506 0.0347405374]
     [0.0357381776 0.0707135797 0.038807828 ... 0.0271954089 0.00653049164 0.0103892926]
     ...
     [0.0371701233 0.0127153378 0.0391794108 ... 0.0263294745 0.0337658711 0.0299302209]
     [0.0315495543 0.00748188747 0.0134998737 ... 0.0144523028 0.0586976893 0.0509485416]
     [0.0259084832 0.00705726258 0.0294559952 ... 0.0486883894 0.0732757 0.11500632]]
    Epoch 91 Loss None
    Time taken for 1 epoch 3.3607592582702637 sec
    
    [[0.251330078 0.00821497 0.0780266598 ... 0.0109853717 0.0114870584 0.0116940951]
     [0.0261162724 0.174090281 0.0257762838 ... 0.230390042 0.130956516 0.0388825908]
     [0.0343843475 0.0682729408 0.038337566 ... 0.0266120508 0.00622358359 0.010043324]
     ...
     [0.0359734893 0.0120804813 0.0378091149 ... 0.0253042653 0.0331187546 0.0287522245]
     [0.0310181957 0.00725160912 0.0131340371 ... 0.0140513163 0.0564037375 0.0496945642]
     [0.024867259 0.00677443435 0.0290696658 ... 0.0464710258 0.0717130527 0.110502742]]
    Epoch 92 Loss None
    Time taken for 1 epoch 3.226728916168213 sec
    
    [[0.242750645 0.00795955211 0.075110741 ... 0.0106555317 0.0112529844 0.0113746244]
     [0.0251308344 0.162767604 0.0246232711 ... 0.223687187 0.130461529 0.040036127]
     [0.0335983969 0.0660057813 0.0365125649 ... 0.0256071165 0.00607655197 0.00979203172]
     ...
     [0.0348053575 0.0116400123 0.036653351 ... 0.0242887083 0.0322338156 0.0282179862]
     [0.0310631506 0.00710461 0.0129820174 ... 0.0135150459 0.05483694 0.0479913242]
     [0.0238623153 0.00651379256 0.0279826969 ... 0.0457014665 0.0687291101 0.105806462]]
    Epoch 93 Loss None
    Time taken for 1 epoch 3.2709133625030518 sec
    
    [[0.226934269 0.00770406937 0.0730231851 ... 0.0104456842 0.0110301757 0.0111043332]
     [0.0245494116 0.158077449 0.0242712554 ... 0.207289875 0.118215807 0.0337274894]
     [0.0317642577 0.0658114776 0.0358073041 ... 0.0249005146 0.00586301554 0.0095671257]
     ...
     [0.033653032 0.0115365526 0.035542503 ... 0.0234631058 0.0315883644 0.0271092076]
     [0.0294218473 0.00682996446 0.0123878336 ... 0.0130912103 0.0527509116 0.0466390885]
     [0.0234832503 0.00634441571 0.0271300916 ... 0.0431051925 0.0680597 0.100608326]]
    Epoch 94 Loss None
    Time taken for 1 epoch 3.214726448059082 sec
    
    [[0.217476428 0.00752045959 0.0705760643 ... 0.0100679891 0.0108184023 0.0108110905]
     [0.0239959322 0.150719926 0.023977194 ... 0.199647024 0.112612754 0.0312963314]
     [0.0304185636 0.0643340349 0.0345203653 ... 0.0239615981 0.00566531718 0.00929256529]
     ...
     [0.0326875746 0.0110505726 0.034214098 ... 0.0226558167 0.0309253912 0.0265061818]
     [0.0285148583 0.00667189108 0.0121556204 ... 0.012657661 0.051304996 0.0453475676]
     [0.0224738829 0.00619337335 0.0262188148 ... 0.0424269587 0.0642946884 0.0976969525]]
    Epoch 95 Loss None
    Time taken for 1 epoch 3.2217319011688232 sec
    
    [[0.207895949 0.00729717175 0.0682581291 ... 0.0099742813 0.0104121799 0.010595968]
     [0.0236971322 0.14503178 0.0231936164 ... 0.191862941 0.11076197 0.0326155759]
     [0.0292886067 0.0631307662 0.0338274129 ... 0.0232448652 0.00553077 0.00907002762]
     ...
     [0.0317099802 0.0106216809 0.0334234 ... 0.0219507422 0.0300724 0.0255069509]
     [0.027407106 0.00650360668 0.0118381828 ... 0.0122922231 0.049550157 0.0437601395]
     [0.0218317807 0.00591326365 0.0256495271 ... 0.040398322 0.0641758516 0.0927732438]]
    Epoch 96 Loss None
    Time taken for 1 epoch 3.25073504447937 sec
    
    [[0.199105591 0.00706767943 0.0657487437 ... 0.00959558133 0.0102454685 0.0102937268]
     [0.0226657223 0.136579677 0.0223919377 ... 0.185112655 0.107156672 0.0317334235]
     [0.0282774381 0.0611607805 0.0325026102 ... 0.0225773845 0.0053806724 0.00886622816]
     ...
     [0.0306375567 0.0105406465 0.0323069915 ... 0.0211793464 0.0294135138 0.0250930525]
     [0.0274676494 0.0063370713 0.0114856437 ... 0.0118725812 0.0479942784 0.0427935459]
     [0.0211772453 0.00581489829 0.024825165 ... 0.0398764573 0.0602413937 0.0902323425]]
    Epoch 97 Loss None
    Time taken for 1 epoch 3.2467334270477295 sec
    
    [[0.189151958 0.00690916972 0.0643492341 ... 0.00944255 0.0100022526 0.0100648031]
     [0.0224844906 0.132448867 0.0226682872 ... 0.17871429 0.107398786 0.0317167938]
     [0.0271292794 0.0607266 0.0320179425 ... 0.0218104366 0.0052407505 0.00864607655]
     ...
     [0.0298032966 0.0101312436 0.0313550197 ... 0.0205701627 0.0289131 0.0241239443]
     [0.0264028553 0.00617607636 0.0111564407 ... 0.0115724942 0.0464939 0.0414396338]
     [0.0205800887 0.00562299928 0.0240824 ... 0.0376921408 0.0607065223 0.0862547159]]
    Epoch 98 Loss None
    Time taken for 1 epoch 3.2487332820892334 sec
    
    [[0.191293851 0.00628637057 0.067284286 ... 0.00915531628 0.00966311526 0.009826147]
     [0.0212992 0.12376257 0.0211405978 ... 0.165452868 0.0956813097 0.0277930349]
     [0.0260032639 0.0599068627 0.030990459 ... 0.0211391971 0.00518442038 0.0084916]
     ...
     [0.0289194696 0.00975201279 0.0305812526 ... 0.019906152 0.0279808417 0.0238060933]
     [0.0260376446 0.00606328156 0.0109185167 ... 0.0112084271 0.045271121 0.0403177179]
     [0.0198755339 0.00545216724 0.0234574 ... 0.0369951762 0.0568723418 0.0838438645]]
    Epoch 99 Loss None
    Time taken for 1 epoch 3.26411509513855 sec
    
    [[0.173976496 0.00656081 0.0604686551 ... 0.0090728635 0.00951965898 0.00958967768]
     [0.021215409 0.121247977 0.0209491644 ... 0.163445696 0.0966496468 0.0259816609]
     [0.0253606364 0.0575516559 0.0301852953 ... 0.0205024257 0.00508100213 0.00828082487]
     ...
     [0.0280481894 0.00969074201 0.0295664128 ... 0.0192937367 0.0277799331 0.0229165982]
     [0.0240913611 0.00585080869 0.0105555085 ... 0.0108943442 0.0436273143 0.0390870161]
     [0.0195689537 0.00530016 0.0228362028 ... 0.0360033885 0.0564572588 0.0802763328]]
    Epoch 100 Loss None
    Time taken for 1 epoch 3.232729911804199 sec
    
    


```python
# 1000 epoch, batchsize 12, 18234 vocab, appox 20 hours


```

# Predict


```python
tf.train.latest_checkpoint(checkpoint_dir)
#tf.train.load_checkpoint(checkpoint_dir+"/ckpt_150.index")
```




    './training_checkpoints\\ckpt_99'




```python
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))
```


```python
model.summary()
```

    Model: "sequential_2"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_2 (Embedding)      (1, None, 256)            1633536   
    _________________________________________________________________
    gru_2 (GRU)                  (1, None, 256)            394752    
    _________________________________________________________________
    lstm_2 (LSTM)                (1, None, 1024)           5246976   
    _________________________________________________________________
    dense_2 (Dense)              (1, None, 6381)           6540525   
    =================================================================
    Total params: 13,815,789
    Trainable params: 13,815,789
    Non-trainable params: 0
    _________________________________________________________________
    


```python
def predict(model, start_string):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = 300

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.9

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # using a multinomial distribution to predict the word returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted word as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)
       
      text_generated.append(" "+idx2char[predicted_id])

  return (start_string + ''.join(text_generated))
```


```python
model.summary()
```

    Model: "sequential_2"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_2 (Embedding)      (1, None, 256)            1633536   
    _________________________________________________________________
    gru_2 (GRU)                  (1, None, 256)            394752    
    _________________________________________________________________
    lstm_2 (LSTM)                (1, None, 1024)           5246976   
    _________________________________________________________________
    dense_2 (Dense)              (1, None, 6381)           6540525   
    =================================================================
    Total params: 13,815,789
    Trainable params: 13,815,789
    Non-trainable params: 0
    _________________________________________________________________
    


```python
print(predict(model, start_string=u'cr'))
```

    cr career misleadingly explosive attacked could removed book department hit problem already younger gone steady chance biggest want initiative world plant job together see song n old hawaii meltdown mother loyal video dr company photoscan attack contestant visit exercise jail like become reward drive people buy threat try including parabo see researcher department son meal mr cahill well later author taking directed told crowd collection state officiated likely camera administration people jump phase least hallway penalty want clock mu calling mean happened island stop age reagan became cower several calorie requires hard finding certain quarterback split outside friend probably arnold petraeus others group family avoid u dispute decisive day apartment kim fernandez would threat environmental november nrg selected remain veteran note detention job given cue northeast used exactly health wa started remains ignore housing show passed alongside senselessness clip smaller consume district run drone earn trust department reach said military made science got shame state penalty george night drug allowing group culture according burning recuperating ignore see power something also await death workload result officer left question research turn matter prisoner hold shift later iraq priority instructed give meet part evening researcher friend african depart including personal heiress spot way agreement would understand province right pollute allows last million equipment supposed beating islamic state station wong might opposed used three bronx month p scientific area official course force american community sent could detective arrival general defendant hardwood committed annual east hour balance come intervention score point least home got nrg lifting president park official plant home mother narrative seen affordable troop wa review wa inside pant first line resorted mr fernandez denmark executive savage ten started home beyond seeking massive war remained allowing high china taliban workout weight pressed lower day reward relation provide choose news throughout note seven death company
    


```python

```
