import pandas as pd
import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize  # Import the word_tokenize function
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import gensim
from gensim import corpora
from gensim.models import LdaModel
from sklearn.feature_extraction.text import CountVectorizer
import string

# Step 1: Read the Excel data
data = pd.read_excel("CleanedTrainingData_mod.xlsx")
#print(data.head(2))
#print(data['tweetcontent'])
# Apply Preprocessing on the Corpus

# stop loss words 
stop = set(stopwords.words('english'))

# Stopwords from stopwords-json
stopwords_json = {"en":["a","a's","able","about","above","according","accordingly","across","actually","after","afterwards","again","against","ain't","all","allow","allows","almost","alone","along","already","also","although","always","am","among","amongst","an","and","another","any","anybody","anyhow","anyone","anything","anyway","anyways","anywhere","apart","appear","appreciate","appropriate","are","aren't","around","as","aside","ask","asking","associated","at","available","away","awfully","b","be","became","because","become","becomes","becoming","been","before","beforehand","behind","being","believe","below","beside","besides","best","better","between","beyond","both","brief","but","by","c","c'mon","c's","came","can","can't","cannot","cant","cause","causes","certain","certainly","changes","clearly","co","com","come","comes","concerning","consequently","consider","considering","contain","containing","contains","corresponding","could","couldn't","course","currently","d","definitely","described","despite","did","didn't","different","do","does","doesn't","doing","don't","done","down","downwards","during","e","each","edu","eg","eight","either","else","elsewhere","enough","entirely","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","exactly","example","except","f","far","few","fifth","first","five","followed","following","follows","for","former","formerly","forth","four","from","further","furthermore","g","get","gets","getting","given","gives","go","goes","going","gone","got","gotten","greetings","h","had","hadn't","happens","hardly","has","hasn't","have","haven't","having","he","he's","hello","help","hence","her","here","here's","hereafter","hereby","herein","hereupon","hers","herself","hi","him","himself","his","hither","hopefully","how","howbeit","however","i","i'd","i'll","i'm","i've","ie","if","ignored","immediate","in","inasmuch","inc","indeed","indicate","indicated","indicates","inner","insofar","instead","into","inward","is","isn't","it","it'd","it'll","it's","its","itself","j","just","k","keep","keeps","kept","know","known","knows","l","last","lately","later","latter","latterly","least","less","lest","let","let's","like","liked","likely","little","look","looking","looks","ltd","m","mainly","many","may","maybe","me","mean","meanwhile","merely","might","more","moreover","most","mostly","much","must","my","myself","n","name","namely","nd","near","nearly","necessary","need","needs","neither","never","nevertheless","new","next","nine","no","nobody","non","none","noone","nor","normally","not","nothing","novel","now","nowhere","o","obviously","of","off","often","oh","ok","okay","old","on","once","one","ones","only","onto","or","other","others","otherwise","ought","our","ours","ourselves","out","outside","over","overall","own","p","particular","particularly","per","perhaps","placed","please","plus","possible","presumably","probably","provides","q","que","quite","qv","r","rather","rd","re","really","reasonably","regarding","regardless","regards","relatively","respectively","right","s","said","same","saw","say","saying","says","second","secondly","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sensible","sent","serious","seriously","seven","several","shall","she","should","shouldn't","since","six","so","some","somebody","somehow","someone","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specified","specify","specifying","still","sub","such","sup","sure","t","t's","take","taken","tell","tends","th","than","thank","thanks","thanx","that","that's","thats","the","their","theirs","them","themselves","then","thence","there","there's","thereafter","thereby","therefore","therein","theres","thereupon","these","they","they'd","they'll","they're","they've","think","third","this","thorough","thoroughly","those","though","three","through","throughout","thru","thus","to","together","too","took","toward","towards","tried","tries","truly","try","trying","twice","two","u","un","under","unfortunately","unless","unlikely","until","unto","up","upon","us","use","used","useful","uses","using","usually","uucp","v","value","various","very","via","viz","vs","w","want","wants","was","wasn't","way","we","we'd","we'll","we're","we've","welcome","well","went","were","weren't","what","what's","whatever","when","whence","whenever","where","where's","whereafter","whereas","whereby","wherein","whereupon","wherever","whether","which","while","whither","who","who's","whoever","whole","whom","whose","why","will","willing","wish","with","within","without","won't","wonder","would","wouldn't","x","y","yes","yet","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","z","zero"]}
stop = stop.union(set(stopwords_json['en']))

# punctuation 
exclude = set(string.punctuation) 
# exclude = exclude.union(set(top_k_words))

# lemmatization
lemma = WordNetLemmatizer() 
porter = PorterStemmer()

# One function for all the steps:
def clean(doc):

# convert text into lower case + split into words
    pattern = r'[^a-zA-Z0-9\s]'
    stop_free = " ".join([re.sub(pattern, '', i) for i in doc.lower().split() if i not in stop])

# remove any stop words present
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)

# remove punctuations + normalize the text
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())

#stemming
    stemmed = " ".join(porter.stem(word) for word in normalized.split())

    return stemmed  

corpus = data["tweetcontent"]
str_corpus = [str(doc) for doc in corpus] 
clean_corpus = [clean(doc).split() for doc in str_corpus]
#print(clean_corpus)

'''
# Step 2: Preprocess the text
# Convert tweet content to string and tokenize
data['tweetcontent'] = data['tweetcontent'].astype(str).apply(word_tokenize)

# Remove stopwords
stop_words = set(stopwords.words('english'))
data['tweetcontent'] = data['tweetcontent'].apply(lambda x: [word for word in x if word not in stop_words])

# Lemmatize the words
lemmatizer = WordNetLemmatizer()
data['tweetcontent'] = data['tweetcontent'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

# Convert list of tokens back to string
data['tweetcontent'] = data['tweetcontent'].apply(' '.join)

print(data)
'''

#tokenized_text = clean_corpus.apply(lambda x: x.split()).tolist()
tokenized_text = clean_corpus

# Create a dictionary and a corpus
dictionary = corpora.Dictionary(tokenized_text)
corpus = [dictionary.doc2bow(text) for text in tokenized_text]

print("enter the number of classifications you want")
x = input()

num_topics = int(x)  # Adjust this number based on your data and goals
# Train the LDA model
lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42)

for idx, topic in lda_model.print_topics(num_words=4):
    print(f"Topic #{idx}: {topic}")
#Next Steps: 
#Feed the pre-processed data to the model
#Train an existing model on our pre-processed data
#Use that model to get ouputs as needed
# Get the dominant topic for each document
dominant_topics = []

for doc in corpus:
    topic_probs = lda_model[doc]
    # Get the topic with the highest probability
    dominant_topic = max(topic_probs, key=lambda x: x[1])[0]
    dominant_topics.append(dominant_topic)

# Add the dominant topic information to the DataFrame
data['dominant_topic'] = dominant_topics
grouped = data.groupby('username')

# Create a dictionary with usernames as keys and lists of unique topics as values
user_topics_dict = {name: list(set(group['dominant_topic'])) for name, group in grouped}

# Filter the dictionary to keep only entries with more than one topic
user_topics_dict_filtered = {k: v for k, v in user_topics_dict.items() if len(v) > 2}

# Print the filtered dictionary
print(user_topics_dict_filtered)
