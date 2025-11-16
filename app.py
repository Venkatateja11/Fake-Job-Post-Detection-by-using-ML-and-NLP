from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)
app.secret_key = '29dd200ff739e8e141581ce611a8fca2'
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')


UPLOAD_FOLDER = './dataset'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/load_data', methods=['GET', 'POST'])
def load_data():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            flash('File successfully uploaded')
            return redirect(url_for('load_data'))
    return render_template('load_data.html')


@app.route('/preprocess', methods=['GET', 'POST'])
def preprocess():
    if request.method == 'POST':
        # Load the dataset
        try:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], os.listdir(app.config['UPLOAD_FOLDER'])[0])
            data = pd.read_csv(file_path)

            # Preprocess steps (example: splitting into train/test)
            train = data.sample(frac=0.8, random_state=42)
            test = data.drop(train.index)

            # Save the processed files
            train.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'train.csv'), index=False)
            test.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'test.csv'), index=False)

            flash('Data preprocessing completed. Train/Test sets created.')
        except Exception as e:
            flash(f'Error during preprocessing: {e}')

        return redirect(url_for('preprocess'))

    return render_template('preprocess.html')




@app.route('/visualizations')
def visualizations():
    try:
        # Load preprocessed dataset
        train_path = os.path.join(app.config['UPLOAD_FOLDER'], 'train.csv')
        train_data = pd.read_csv(train_path)

        # Combine columns into a single 'text' column for word clouds
        columns_to_combine = ['title', 'location', 'company_profile', 'description', 'requirements', 'benefits', 'industry']
        for col in columns_to_combine:
            train_data[col] = train_data[col].fillna('')

        train_data['text'] = train_data['title'] + ' ' + train_data['location'] + ' ' + \
                             train_data['company_profile'] + ' ' + train_data['description'] + ' ' + \
                             train_data['requirements'] + ' ' + train_data['benefits'] + ' ' + train_data['industry']

        # Generate Word Clouds
        fake_jobs = train_data[train_data['fraudulent'] == 1]['text']
        real_jobs = train_data[train_data['fraudulent'] == 0]['text']

        fake_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(fake_jobs))
        real_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(real_jobs))

        # Save word clouds
        os.makedirs('./static/images', exist_ok=True)
        fake_wordcloud.to_file('./static/images/fake_wordcloud.png')
        real_wordcloud.to_file('./static/images/real_wordcloud.png')

        # Generate Additional Visualizations
        plt.figure(figsize=(10, 6))
        sns.countplot(x='fraudulent', data=train_data)
        plt.title('Distribution of Fake vs Real Jobs')
        plt.savefig('./static/images/job_distribution.png')

        plt.figure(figsize=(10, 6))
        train_data['location'].fillna('Unknown').value_counts().head(10).plot(kind='bar')
        plt.title('Top 10 Locations in Job Postings')
        plt.ylabel('Number of Postings')
        plt.savefig('./static/images/top_locations.png')

        plt.figure(figsize=(10, 6))
        train_data['industry'].fillna('Unknown').value_counts().head(10).plot(kind='barh')
        plt.title('Top 10 Industries in Job Postings')
        plt.xlabel('Number of Postings')
        plt.savefig('./static/images/top_industries.png')

        plt.figure(figsize=(10, 6))
        train_data['title'].fillna('Unknown').value_counts().head(10).plot(kind='bar')
        plt.title('Top 10 Job Titles')
        plt.ylabel('Number of Postings')
        plt.savefig('./static/images/top_titles.png')

        plt.figure(figsize=(10, 6))
        train_data['company_profile'].fillna('').str.len().plot(kind='hist', bins=20)
        plt.title('Company Profile Length Distribution')
        plt.xlabel('Character Count')
        plt.savefig('./static/images/company_profile_length.png')

        return render_template('visualizations.html')

    except Exception as e:
        flash(f"Error generating visualizations: {e}")
        return redirect(url_for('preprocess'))




from sklearn.metrics import classification_report

@app.route('/model', methods=['GET', 'POST'])
def model():
    if request.method == 'POST':
        try:
            # Load the training dataset
            train_path = os.path.join(app.config['UPLOAD_FOLDER'], 'train.csv')
            train_data = pd.read_csv(train_path)

            # Combine columns into a single 'text' column
            columns_to_combine = ['title', 'location', 'company_profile', 'description', 'requirements', 'benefits', 'industry']
            for col in columns_to_combine:
                train_data[col] = train_data[col].fillna('')

            train_data['text'] = train_data['title'] + ' ' + train_data['location'] + ' ' + \
                                 train_data['company_profile'] + ' ' + train_data['description'] + ' ' + \
                                 train_data['requirements'] + ' ' + train_data['benefits'] + ' ' + train_data['industry']

            # Features and target
            X = train_data['text']
            y = train_data['fraudulent']  # Replace with your target column name

            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

            # Vectorize the text data
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=1000)
            X_train_vectorized = vectorizer.fit_transform(X_train)
            X_val_vectorized = vectorizer.transform(X_val)

            # Train Naive Bayes model
            from sklearn.naive_bayes import MultinomialNB
            nb_model = MultinomialNB()
            nb_model.fit(X_train_vectorized, y_train)
            nb_preds = nb_model.predict(X_val_vectorized)
            nb_report = classification_report(y_val, nb_preds, output_dict=True)

            # Train Decision Tree model
            from sklearn.tree import DecisionTreeClassifier
            dt_model = DecisionTreeClassifier(random_state=42)
            dt_model.fit(X_train_vectorized, y_train)
            dt_preds = dt_model.predict(X_val_vectorized)
            dt_report = classification_report(y_val, dt_preds, output_dict=True)

            # Prepare classification reports for display
            nb_report_table = pd.DataFrame(nb_report).transpose().round(2).to_html(classes="table table-striped", justify="center")
            dt_report_table = pd.DataFrame(dt_report).transpose().round(2).to_html(classes="table table-striped", justify="center")

            return render_template('model.html', nb_report_table=nb_report_table, dt_report_table=dt_report_table)

        except Exception as e:
            flash(f"Error during model training: {e}")

        return redirect(url_for('model'))

    return render_template('model.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Retrieve user input
            job_description = request.form['job_description']

            # Validate input
            if not isinstance(job_description, str) or not job_description.strip():
                flash("Invalid input. Please provide a valid job description.")
                return redirect(url_for('predict'))

            # Load the training dataset
            train_path = os.path.join(app.config['UPLOAD_FOLDER'], 'train.csv')
            train_data = pd.read_csv(train_path)

            # Handle missing values and combine columns into 'text'
            columns_to_combine = ['title', 'location', 'company_profile', 'description', 'requirements', 'benefits', 'industry']
            for col in columns_to_combine:
                train_data[col] = train_data[col].fillna('')

            train_data['text'] = train_data['title'] + ' ' + train_data['location'] + ' ' + \
                                 train_data['company_profile'] + ' ' + train_data['description'] + ' ' + \
                                 train_data['requirements'] + ' ' + train_data['benefits'] + ' ' + train_data['industry']

            # Extract features and target
            X = train_data['text']
            y = train_data['fraudulent']  # Replace with your actual target column name

            # Preprocess input data
            input_data = pd.DataFrame([job_description], columns=['text'])

            # Transform text data using TfidfVectorizer
            from sklearn.feature_extraction.text import TfidfVectorizer

            vectorizer = TfidfVectorizer(max_features=1000)
            X_vectorized = vectorizer.fit_transform(X)
            input_vectorized = vectorizer.transform(input_data['text'])

            # Train a Naive Bayes model (or load a pre-trained model)
            nb_model = GaussianNB()
            nb_model.fit(X_vectorized.toarray(), y)
            prediction = nb_model.predict(input_vectorized.toarray())

            # Flash the result
            result = "Fake Job Posting" if prediction[0] == 1 else "Genuine Job Posting"
            flash(f"Prediction: {result}")

        except Exception as e:
            flash(f"Error during prediction: {e}")

        return redirect(url_for('predict'))

    return render_template('prediction.html')




if __name__ == '__main__':
    app.run(debug=True)
