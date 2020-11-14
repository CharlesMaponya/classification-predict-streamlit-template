"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os
import nltk
import string
import re
#import contractions
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.probability import FreqDist
from wordcloud import WordCloud, ImageColorGenerator
import warnings
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

warnings.filterwarnings(action = 'ignore') 

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
st.set_option('deprecation.showPyplotGlobalUse', False)

# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")
retweet = 'RT'
import streamlit.components.v1 as components

st.cache(suppress_st_warning=True,allow_output_mutation=True)
def class_analysis(df):
    df['sent_labels']  = df['sentiment'].map({-1: 'Anti',0:'Neutral', 1:'Pro', 2:'News'})
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(20, 10), dpi=100)
    
    sns.countplot(df['sent_labels'], ax=axes[0])
    code_labels=['Pro', 'News', 'Neutral', 'Anti']
    axes[1].pie(df['sent_labels'].value_counts(),labels= code_labels,autopct='%1.0f%%',startangle=90,explode = (0.1, 0.1, 0.1, 0.1))
    fig.suptitle('Sentiment Class Analysis', fontsize=20)
    st.pyplot(fig)

st.cache(suppress_st_warning=True,allow_output_mutation=True)
def class_dist(df):
    df['sent_labels']  = df['sentiment'].map({-1: 'Anti',0:'Neutral', 1:'Pro', 2:'News'})
    df['text_length'] = df['message'].apply(lambda x: len(x))
    fig, axis = plt.subplots(ncols=2,nrows=1, dpi=100)
    
    sns.boxplot(x=df['sent_labels'],y=df['text_length'],data=df,ax=axis[0],color = 'orange')

    sns.violinplot(x=df['sent_labels'], y=df['text_length'],ax=axis[1])
    plt.xlabel('Sentiment Class')
    plt.ylabel('Tweet Length')
    plt.tight_layout()
    st.pyplot(fig)


st.cache(suppress_st_warning=True,allow_output_mutation=True)
def mentions(x):
    x = re.sub(r"(?:\@|https?\://)\S+", "", x)
    return x

st.cache(suppress_st_warning=True,allow_output_mutation=True)
def remove_punc(x):
    x = re.sub(r"([^A-Za-z0-9]+)"," ",x)
    return x

st.cache(suppress_st_warning=True,allow_output_mutation=True)
def StopWords():
    stop_words = set(stopwords.words('english'))
    return stop_words

st.cache(suppress_st_warning=True,allow_output_mutation=True)
def word_count(train):
    cnt = Counter()
    for message in train['message'].values:
        for word in message:
            cnt[word] +=1
    return cnt.most_common(20)

st.cache(suppress_st_warning=True,allow_output_mutation=True)
def data_cleaning(df):
    wnl = WordNetLemmatizer()
    df['message'] = df['message'].apply(mentions)
    #df['message'] = df['message'].apply(lambda x: contractions.fix(x))
    df['message'] = df['message'].str.replace(r"http\S+|www.\S+", "", case=False)
    df['message'] = df['message'].map(lambda x: remove_punc(str(x)))
    df['message'] = df['message'].apply(word_tokenize)
    df['message'] = df['message'].apply(lambda x: [word for word in x if word not in retweet])
    df['message'] = df['message'].apply(lambda x : [word.lower() for word in x])
    df['message'] = df['message'].apply(lambda x: [word for word in x if word not in StopWords()])
    df['pos_tags'] = df['message'].apply(nltk.tag.pos_tag)
    df['wordnet_pos'] = df['pos_tags'].apply(lambda x: [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in x])
    df['message'] = df['wordnet_pos'].apply(lambda x: [wnl.lemmatize(word, tag) for word, tag in x])
    return df

st.cache(suppress_st_warning=True,allow_output_mutation=True)
def pro_mostpopular(df):
    pro_popular = df[df['sentiment'] == 1]
    pro_pop = word_count(pro_popular)
    return pro_pop

st.cache(suppress_st_warning=True,allow_output_mutation=True)
def anti_mostpopular(df):
    anti_popular = df[df['sentiment']== -1]
    anti_pop = word_count(anti_popular)
    return anti_pop

st.cache(suppress_st_warning=True,allow_output_mutation=True)
def neutral_mostpopular(df):
    neutral = df[df['sentiment']==0]
    neutral_pop = word_count(neutral)
    return neutral_pop

st.cache(suppress_st_warning=True,allow_output_mutation=True)
def news_mostpopular(df):
    news = df[df['sentiment']==2]
    news_pop = word_count(news)
    return news_pop

st.cache(suppress_st_warning=True,allow_output_mutation=True)
def popularwords_visualizer(data):
    news = news_mostpopular(data)
    pro = pro_mostpopular(data)
    anti=anti_mostpopular(data)
    neutral = neutral_mostpopular(data)
    
    #Creating the Subplots for Most Popular words
    fig, axs = plt.subplots(2, 2)
    
    plt.setp(axs[-1, :], xlabel='Most popular word (Descending)')
    plt.setp(axs[:, 0], ylabel='# of times the word appeard')
    axs[0,0].bar(range(len(news)),[val[1] for val in news],align='center')
    axs[0,0].set_xticks(range(len(news)), [val[0] for val in news])
    axs[0,0].set_title("News Class")
    
    axs[0,1].bar(range(len(neutral)),[val[1] for val in neutral],align='center')
    axs[0,1].set_xticks(range(len(neutral)), [val[0] for val in neutral])
    axs[0,1].set_title("Neutral Class")
    
    axs[1,0].bar(range(len(pro)),[val[1] for val in pro],align='center')
    axs[1,0].set_xticks(range(len(pro)), [val[0] for val in pro])
    axs[1,0].set_title("Pro Class")
    
    axs[1,1].bar(range(len(anti)),[val[1] for val in anti],align='center')
    axs[1,1].set_xticks(range(len(anti)), [val[0] for val in anti])
    axs[1,1].set_title("Anti Class")
    fig.tight_layout()
    st.pyplot(fig)

st.cache(suppress_st_warning=True,allow_output_mutation=True)
def wordcloud_visualizer(df):
    news = df['message'][df['sentiment']==2].str.join(' ')
    neutral = df['message'][df['sentiment']==2].str.join(' ')
    pro = df['message'][df['sentiment']==2].str.join(' ')
    anti = df['message'][df['sentiment']==2].str.join(' ')
    #Visualize each sentiment class
    fig, axis = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))
    news_wordcloud = WordCloud(width=900, height=600, background_color='white', colormap='winter').generate(str(news))
    axis[0, 0].imshow(news_wordcloud)
    axis[0, 0].set_title('News Class',fontsize=14)
    axis[0, 0].axis("off") 
    neutral_wordcloud = WordCloud(width=900, height=600, background_color='white', colormap='winter', min_font_size=10).generate(str(neutral))
    axis[1, 0].imshow(neutral_wordcloud)
    axis[1, 0].set_title('Neutral Class',fontsize=14)
    axis[1, 0].axis("off") 
    
    pro_wordcloud = WordCloud(width=900, height=600, background_color='white', colormap='winter', min_font_size=10).generate(str(pro))
    axis[0, 1].imshow(pro_wordcloud)
    axis[0, 1].set_title('Pro Class',fontsize=14)
    axis[0, 1].axis("off") 
    anti_wordcloud = WordCloud(width=900, height=600, background_color='white', colormap='winter', min_font_size=10).generate(str(anti))
    axis[1, 1].imshow(anti_wordcloud)
    axis[1, 1].set_title('Anti Class',fontsize=14)
    axis[1, 1].axis("off")
    st.pyplot(fig)


def prediction_output(predict):
    if predict[0]==-1:
        output="Anti"
        st.error("Text Categorized as: {}".format(output))
    elif predict[0]==0:
        output="Neutral"
        st.info("Text Categorized as: {}".format(output))
    elif predict[0]==1:
        output ="Pro"
        st.success("Text Categorized as: {}".format(output))
    else:
        output = "News"
        st.warning("Text Categorized as: {}".format(output))

st.cache(suppress_st_warning=True,allow_output_mutation=True)
def markup(selection):
    html_temp = """<div style="background-color:{};padding:10px;border-radius:10px; margin-bottom:15px;"><h1 style="color:{};text-align:center;">"""+selection+"""</h1></div>"""
    st.markdown(html_temp, unsafe_allow_html=True)

st.cache(suppress_st_warning=True,allow_output_mutation=True)
def title_tag(title):
    html_temp = """<div style="background-color:{};padding:10px;border-radius:10px; margin-bottom:15px;"><h2 style="color:#00ACEE;text-align:center;">"""+title+"""</h2></div>"""
    st.markdown(html_temp, unsafe_allow_html=True)

#Getting the WordNet Parts of Speech
st.cache(suppress_st_warning=True,allow_output_mutation=True)
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN    
# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["About Predict","Text Classification","Exploratory Data Analysis","Model Performance Evaluation","Our Team"]
	selection = st.sidebar.selectbox("Choose Option", options)
	# Building out the "Information" page
	if selection == "About Predict":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Text Classification":
		markup(selection)
		# Creating a text box for user input
		models = ["Support Vector Classifier","Logisitic Regression Classifier","Optimized Support Vector Classifier"]
		modeloptions = st.selectbox("Choose Predictive Classification Model",models)
		if modeloptions == "Support Vector Classifier":
			tweet_text = st.text_area("Enter Text","Type Here")
			if st.button("Predict text class with Support Vector Classifier"):
				pred = joblib.load(open(os.path.join("resources/LinearSVC.pkl"),"rb"))
				predict = pred.predict([tweet_text])
				prediction_output(predict)
		elif modeloptions =="Logisitic Regression Classifier":
			logi_text = st.text_area("Enter Text","Type Here")
			if st.button("Predict text class with Logisitic Regression Classifier"):
				pred = joblib.load(open(os.path.join("resources/LogisticReg.pkl"),"rb"))
				predict = pred.predict([logi_text])
				prediction_output(predict)
		elif modeloptions =="Optimized Support Vector Classifier":
			svc_text = st.text_area("Enter Text","Type Here")
			if st.button("Predict text class with Optimized Support Vector Classifier"):
				pred = joblib.load(open(os.path.join("resources/LogisticReg.pkl"),"rb"))
				predict = pred.predict([svc_text])
				prediction_output(predict)
	
	if selection == "Exploratory Data Analysis":
		markup(selection)
		print('....Cleaning the Raw data')
		train = data_cleaning(raw)
		visuals =["Sentiment Class Analysis","Message length for each sentiment class","Popular Words Analysis","Word Cloud Analysis"]
		visualselection = st.selectbox("Choose EDA visuals",visuals)

		if visualselection == "Sentiment Class Analysis":
			print('..... Creating the sentiment class analysis visual')
			title_tag("Sentiment Class Analysis")
			class_analysis(train)
		elif visualselection == "Message length for each sentiment class":
			print('...... Creating the sentiment class message length visual')
			title_tag('Message length for each sentiment class')
			class_dist(train)
		elif visualselection =="Popular Words Analysis":
			print('...... Creating the popular words visual')
			title_tag("Popular Words Analysis")
			popularwords_visualizer(train)
		elif visualselection == "Word Cloud Analysis":
			print('..... Creating the WordClouds for sentiment classes')
			title_tag("Word Cloud Analysis")
			wordcloud_visualizer(train)

	if selection == "Model Performance Evaluation":
		title_tag(selection)
# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
