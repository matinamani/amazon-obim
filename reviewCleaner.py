from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import string as str
import re


def cleaner(text):
    stop = set(stopwords.words('english'))
    punctuations = set(str.punctuation)
    lemmatizer = WordNetLemmatizer()

    text = text.lower()
    text = text.strip()
    text = remove_emojis(text)

    words = text.split()

    # removes punctuations
    words = ["".join(x for x in word if (x == "'") | (x not in punctuations))
             for word in words]

    # remove stopwords
    words = [word for word in words if word not in stop]

    # lemmatize words
    words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(word for word in words)


def remove_emojis(text):
    emoji = re.compile("["
                       u"\U0001F600-\U0001F64F"
                       u"\U0001F300-\U0001F5FF"
                       u"\U0001F680-\U0001F6FF"
                       u"\U0001F1E0-\U0001F1FF"
                       u"\U00002500-\U00002BEF"
                       u"\U00002702-\U000027B0"
                       u"\U00002702-\U000027B0"
                       u"\U000024C2-\U0001F251"
                       u"\U0001f926-\U0001f937"
                       u"\U00010000-\U0010ffff"
                       u"\u2640-\u2642"
                       u"\u2600-\u2B55"
                       u"\u200d"
                       u"\u23cf"
                       u"\u23e9"
                       u"\u231a"
                       u"\ufe0f"
                       u"\u3030"
                       "]+", re.UNICODE)
    return re.sub(emoji, '', text)
