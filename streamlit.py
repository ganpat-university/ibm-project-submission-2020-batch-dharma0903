import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
from tld import get_tld
from googlesearch import search
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns  
import streamlit as st
from io import BytesIO

# Read the dataset
df = pd.read_csv('malicious_phish.csv')

# Function to check if URL has an IP address
def having_ip_address(url):
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)' # IPv4 in hexadecimal
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  # Ipv6
    if match:
        return 1
    else:
        return 0
    
# Function to check abnormal URL
def abnormal_url(url):
    hostname = urlparse(url).hostname
    hostname = str(hostname)
    match = re.search(hostname, url)
    if match:
        return 1
    else:
        return 0

# Function to check Google index
def google_index(url):
    site = search(url, 5)
    return 1 if site else 0

# Feature Engineering
def feature_engineering(df):
    df['use_of_ip'] = df['url'].apply(lambda i: having_ip_address(i))
    df['abnormal_url'] = df['url'].apply(lambda i: abnormal_url(i))
    df['google_index'] = df['url'].apply(lambda i: google_index(i))
    df['count.'] = df['url'].apply(lambda i: i.count('.'))
    df['count-www'] = df['url'].apply(lambda i: i.count('www'))
    df['count@'] = df['url'].apply(lambda i: i.count('@'))
    df['count_dir'] = df['url'].apply(lambda i: urlparse(i).path.count('/'))
    df['count_embed_domian'] = df['url'].apply(lambda i: urlparse(i).path.count('//'))
    df['short_url'] = df['url'].apply(lambda i: 1 if re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                                                               'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                                                               'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                                                               'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                                                               'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                                                               'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                                                               'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                                                               'tr\.im|link\.zip\.net', i) else 0)
    df['count-https'] = df['url'].apply(lambda i: i.count('https'))
    df['count-http'] = df['url'].apply(lambda i: i.count('http'))
    df['count%'] = df['url'].apply(lambda i: i.count('%'))
    df['count?'] = df['url'].apply(lambda i: i.count('?'))
    df['count-'] = df['url'].apply(lambda i: i.count('-'))
    df['count='] = df['url'].apply(lambda i: i.count('='))
    df['url_length'] = df['url'].apply(lambda i: len(str(i)))
    df['hostname_length'] = df['url'].apply(lambda i: len(urlparse(i).netloc))
    df['sus_url'] = df['url'].apply(lambda i: 1 if re.search('PayPal|login|signin|bank|account|update|free|lucky|service|bonus|ebayisapi|webscr', i) else 0)
    df['count-digits'] = df['url'].apply(lambda i: sum(c.isdigit() for c in i))
    df['count-letters'] = df['url'].apply(lambda i: sum(c.isalpha() for c in i))
    df['fd_length'] = df['url'].apply(lambda i: len(urlparse(i).path.split('/')[1]) if urlparse(i).path != '' and len(urlparse(i).path.split('/')) > 1 else 0)
    df['tld'] = df['url'].apply(lambda i: get_tld(i, fail_silently=True))
    df['tld_length'] = df['tld'].apply(lambda i: len(i) if i else -1)
    df.drop("tld", axis=1, inplace=True)
    return df

# Preprocessing and Feature Engineering
def preprocessing(df):
    # Encode the target variable
    lb_make = LabelEncoder()
    df["type_code"] = lb_make.fit_transform(df["type"])

    # Feature Engineering
    df = feature_engineering(df)

    # Predictor Variables
    X = df[['use_of_ip', 'abnormal_url', 'count.', 'count-www', 'count@',
            'count_dir', 'count_embed_domian', 'short_url', 'count-https',
            'count-http', 'count%', 'count?', 'count-',             'count=', 'url_length',
            'hostname_length', 'sus_url', 'fd_length', 'tld_length', 'count-digits',
            'count-letters']]

    # Target Variable
    y = df['type_code']

    return X, y

# Model Building
def build_model(X_train, y_train):
    # 1. Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, max_features='sqrt')
    rf.fit(X_train, y_train)

    # 2. LightGBM Classifier
    lgb = LGBMClassifier(objective='multiclass', boosting_type='gbdt', n_jobs=5,
                         silent=True, random_state=5)
    lgb.fit(X_train, y_train)

    # 3. XGBoost Classifier
    xgb_c = xgb.XGBClassifier(n_estimators=100)
    xgb_c.fit(X_train, y_train)

    return rf, lgb, xgb_c

# Model Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    st.write(classification_report(y_test, y_pred, target_names=['benign', 'defacement', 'phishing', 'malware']))
    score = accuracy_score(y_test, y_pred)
    st.write("Accuracy:   %0.3f" % score)
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=['benign', 'defacement', 'phishing', 'malware'],
                         columns=['benign', 'defacement', 'phishing', 'malware'])
    st.write('Confusion Matrix')
    st.write(cm_df)

# Prediction
def predict_url(url, model):
    status = []
    status.append(having_ip_address(url))
    status.append(abnormal_url(url))
    status.append(url.count('.'))
    status.append(url.count('www'))
    status.append(url.count('@'))
    status.append(urlparse(url).path.count('/'))
    status.append(urlparse(url).path.count('//'))
    status.append(1 if re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                                 'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                                 'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                                 'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                                 'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                                 'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                                 'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                                 'tr\.im|link\.zip\.net', url) else 0)
    status.append(url.count('https'))
    status.append(url.count('http'))
    status.append(url.count('%'))
    status.append(url.count('?'))
    status.append(url.count('-'))
    status.append(url.count('='))
    status.append(len(str(url)))
    status.append(len(urlparse(url).netloc))
    status.append(1 if re.search('PayPal|login|signin|bank|account|update|free|lucky|service|bonus|ebayisapi|webscr', url) else 0)
    status.append(sum(c.isdigit() for c in url))
    status.append(sum(c.isalpha() for c in url))
    status.append(len(urlparse(url).path.split('/')[1]) if urlparse(url).path != '' else 0)
    tld = get_tld(url, fail_silently=True)
    status.append(len(tld) if tld else -1)
    features = np.array(status).reshape((1, -1))
    pred = model.predict(features)
    if int(pred[0]) == 0:
        return "SAFE"
    elif int(pred[0]) == 1:
        return "DEFACEMENT"
    elif int(pred[0]) == 2:
        return "PHISHING"
    elif int(pred[0]) == 3:
        return "MALWARE"

# Function to visualize features of a given URL
# Function to visualize features of a given URL
def visualize_url_features(url):
    status = []
    status.append(having_ip_address(url))
    status.append(abnormal_url(url))
    status.append(url.count('.'))
    status.append(url.count('www'))
    status.append(url.count('@'))
    status.append(urlparse(url).path.count('/'))
    status.append(urlparse(url).path.count('//'))
    status.append(1 if re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                                 'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                                 'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                                 'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                                 'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                                 'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                                 'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                                 'tr\.im|link\.zip\.net', url) else 0)
    status.append(url.count('https'))
    status.append(url.count('http'))
    status.append(url.count('%'))
    status.append(url.count('?'))
    status.append(url.count('-'))
    status.append(url.count('='))
    status.append(len(str(url)))
    status.append(len(urlparse(url).netloc))
    status.append(1 if re.search('PayPal|login|signin|bank|account|update|free|lucky|service|bonus|ebayisapi|webscr', url) else 0)
    status.append(sum(c.isdigit() for c in url))
    status.append(sum(c.isalpha() for c in url))
    status.append(len(urlparse(url).path.split('/')[1]) if urlparse(url).path != '' else 0)
    tld = get_tld(url, fail_silently=True)
    status.append(len(tld) if tld else -1)
    features = np.array(status).reshape((1, -1))

    # Plotting
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    sns.barplot(y=['having_ip_address', 'abnormal_url', 'count.', 'count-www', 'count@',
                   'count_dir', 'count_embed_domian', 'short_url', 'count-https',
                   'count-http', 'count%', 'count?', 'count-', 'count=', 'url_length',
                   'hostname_length', 'sus_url', 'fd_length', 'tld_length', 'count-digits',
                   'count-letters'], x=status, ax=axes[0])
    axes[0].set_title('Features of the URL')  # Set title on the first axes
    axes[0].set_xlabel('Value')
    axes[0].tick_params(axis='x', rotation=90)  # Rotate x-axis labels
    axes[1].set_axis_off()  # Turn off the second axes
    plt.tight_layout()
    st.pyplot(fig)


# Streamlit GUI function
def main():
    # Preprocessing
    X, y = preprocessing(df)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, shuffle=True, random_state=5)

    # Build models
    rf_model, lgb_model, xgb_model = build_model(X_train, y_train)

    # Streamlit GUI
    st.title("URL Safety Checker")
    st.write("Enter URL:")
    url = st.text_input("URL")

    if st.button("Check URL"):
        result_rf = predict_url(url, rf_model)
        result_lgb = predict_url(url, lgb_model)
        result_xgb = predict_url(url, xgb_model)
        st.write(f"The URL '{url}' is predicted to be {result_rf} (Random Forest)")
        st.write(f"The URL '{url}' is predicted to be {result_lgb} (LightGBM)")
        st.write(f"The URL '{url}' is predicted to be {result_xgb} (XGBoost)")

        # Visualize features of the URL
        visualize_url_features(url)

       # Additional visualizations
        st.write("Additional Visualizations:")
        
        # Plotting count of each type of URL
        fig_count_type = plt.figure(figsize=(10, 6))
        sns.countplot(x="type", data=df)
        plt.title("Count of Each Type of URL")
        plt.xticks(rotation=45)
        st.pyplot(fig_count_type)


        # Generate word cloud of URLs
        text = ' '.join(df['url'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig_wordcloud = plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title("Word Cloud of URLs")
        plt.axis('off')
        st.pyplot(fig_wordcloud)

if __name__ == "__main__":
    main()
