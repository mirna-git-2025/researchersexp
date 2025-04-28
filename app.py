from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory, flash
import os
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from flask_pymongo import PyMongo
from bson import ObjectId
from datetime import datetime
from flask import abort
from flask_mail import Mail, Message
import os

from flask  import session 
from flask_bcrypt import Bcrypt
from flask_session import Session

from transformers import DistilBertTokenizer, DistilBertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from flask import  send_file
from transformers import pipeline
import pandas as pd

from datetime import datetime




app = Flask(__name__)
# Load the T5 question generation model
qg_pipeline = pipeline("text2text-generation", model="iarfmoose/t5-base-question-generator")


# Configure Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'moukhtarmirna@gmail.com'
app.config['MAIL_PASSWORD'] = 'vnna fhag votc fmsn'  # Use App Password, NOT your real password
app.config['MAIL_DEFAULT_SENDER'] = 'moukhtarmirna@gmail.com'
mail = Mail(app) 

# Configure session
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

bcrypt = Bcrypt(app)

# --- DIRECT MONGO CONNECTION --- (Replace with your actual URI)
mongo_uri = "mongodb+srv://flaskuser:0000@cluster0.axlhy4c.mongodb.net/Researchersedst?retryWrites=true&w=majority"
client = MongoClient(mongo_uri)


db = client["Researchersedst"]
collection = db["experiencesmk"]


app.secret_key = "secret_key"
users = db["users"]

#ner 
import spacy
import os
import glob
import pandas as pd
import gdown
#####
# Define model directory and Google Drive file ID
model_dir = "en_ner_bc5cdr_md"
file_id = "1adqC_wWGTlEcSRb68XIHkiU7Y8qaCpCV"  # Replace with your actual Google Drive file ID
model_url = f"https://drive.google.com/uc?id={file_id}"

# Check if the model is already downloaded
if not os.path.exists(model_dir):
    print(f"Model not found. Downloading {model_dir}...")
    
    # Use gdown to download the model from Google Drive using the file ID
    gdown.download(model_url, output=f"{model_dir}.whl", quiet=False)
    print(f"{model_dir} downloaded successfully.")

    # Install the downloaded model
    os.system(f"pip install {model_dir}.whl")
else:
    print(f"Model {model_dir} already exists. Skipping download.")
###
# Load the en_ner_bc5cdr_md model
nlp = spacy.load("en_ner_bc5cdr_md")

# Define the folder where the text files are stored
folder_path = "medtext"  # Change this to your folder path

# Function to read text files from the folder
def read_text_files_from_folder(folder_path):
    text_files = glob.glob(os.path.join(folder_path, "*.txt"))
    return text_files

# Function to extract entities (diseases, chemicals) and sentences
def extract_entities_and_sentences(text_files):
    data = []

    # Iterate over each text file
    for file in text_files:
        # Read the content of each file
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()

        # Process the text using the spaCy model
        doc = nlp(text)

        # Extract sentences and related entities
        for sent in doc.sents:
            sentence = sent.text
            diseases = []
            chemicals = []

            # Extract entities within the sentence
            for ent in sent.ents:
                if ent.label_ == "DISEASE":  # Check for diseases
                    diseases.append(ent.text)
                elif ent.label_ == "CHEMICAL":  # Check for chemicals
                    chemicals.append(ent.text)

            # Add the sentence, diseases, chemicals, and filename to the data list
            data.append({
                "Filename": os.path.basename(file),  # Add the filename (without full path)
                "Sentence": sentence,
                "Diseases": ", ".join(diseases),  # Join multiple diseases with a comma
                "Chemicals": ", ".join(chemicals)  # Join multiple chemicals with a comma
            })

    return data


#semantic search
# Load DistilBERT model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

# Function to Generate Embeddings
def generate_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embeddings.numpy().tolist()  # Convert to list for JSON storage
# Function to perform search based on similarity
def executesemanticsearch(query):
    # Get all documents from the collection
    documents = collection.find()

    # Generate the embedding for the query
    query_embedding = generate_embedding(query)

    # List to hold document similarities and texts
    similarities = []
    
    for doc in documents:
        # Use get() to avoid KeyError in case a field is missing
       
        doc_embedding= doc.get('embedding' , '')
        # Compute cosine similarity between query and document embeddings
        similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
        similarities.append((similarity, doc))

    # Sort documents by similarity score (higher is more relevant)
    similarities.sort(reverse=True, key=lambda x: x[0])

    # Get top 5 most relevant results
    results = [sim[1] for sim in similarities[:5]]
    
    return results


# Configuration for file uploads
EXPERIENCE_FOLDER = os.path.join('static', 'experiences')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf', 'mp4', 'avi', 'mkv'}

# Ensure the experiences folder exists
if not os.path.exists(EXPERIENCE_FOLDER):
    os.makedirs(EXPERIENCE_FOLDER)
UPLOAD_FOLDER = "static/experiences"

app.config['UPLOAD_FOLDER'] = EXPERIENCE_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    # Retrieve email from session if available
    email = session.get("email")
    return render_template('index.html', email=email)  # Pass email to the template

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        existing_user = users.find_one({"email": request.form["email"]})

        if existing_user:
            flash("Email already registered. Please log in.", "danger")
            return redirect(url_for("login"))

        hashed_pw = bcrypt.generate_password_hash(request.form["password"]).decode("utf-8")
        users.insert_one({"email": request.form["email"], "password": hashed_pw})

        flash("Registration successful! Please log in.", "success")
        return redirect(url_for("login"))
    if request.method == "GET":

        return render_template("register.html")

# Define the login route
@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        
        # Fetch the user by email
        user =users.find_one({"email": email})
        
        if user:
            # Compare the entered password with the stored hashed password
            if bcrypt.check_password_hash(user["password"], password):
                session["email"] = email  # Store email in session
                flash("Logged in successfully!", "success")
                return redirect(url_for("index"))  # Redirect to index page on successful login
            else:
                flash("Invalid credentials!", "danger")
                return redirect(url_for("login"))  # Redirect to login if password is incorrect
        else:
            flash("User not found!", "danger")
            return redirect(url_for("login"))  # Redirect to login if user doesn't exist
    
    return render_template('login.html')  # Render login form when it's a GET request@app.route("/dashboard")
def dashboard():
    if "username" in session:
        return f"Welcome {session['username']} to the Dashboard"
    else:
        flash("You need to login first!", "warning")
        return redirect(url_for("index"))

@app.route("/logout")
def logout():
    session.pop("email", None)
    flash("Logged out successfully.", "info")
    return redirect(url_for("index"))
@app.route('/run_entity_recognition', methods=['GET', 'POST'])
def run_entity_recognition():
    # Read text files from the folder
    text_files = read_text_files_from_folder(folder_path)

    # Extract entities and sentences
    data = extract_entities_and_sentences(text_files)

    # Create a pandas DataFrame from the extracted data
    df = pd.DataFrame(data)

    # Output to an Excel file
    output_file = "static/nerresult/medical_entities_with_filenames.xlsx"
    df.to_excel(output_file, index=False)
     # Your entity recognition logic here (replace with actual processing)
    message = "✅ Entity recognition completed! Processed file saved."
    return render_template('resultner.html', message=message)
@app.route('/generate_questions', methods=['GET', 'POST'])
def generate_questions():
    questions = []
    file_link = None

    if request.method == 'POST':
        input_text = request.form.get('text')
        if input_text:
            result = qg_pipeline(input_text, max_length=128, do_sample=False)[0]['generated_text']
            questions = result.split('\n') if '\n' in result else [result]

            # Save to Excel
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'generated_questions_{timestamp}.xlsx'
            filepath = os.path.join('static', 'qg_results', filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            pd.DataFrame({'Question': questions}).to_excel(filepath, index=False)

            file_link = filepath

    return render_template('question_form.html', questions=questions, file_link=file_link)
@app.route('/submit_researcher', methods=['POST'])
def submit():
    # Retrieve data from the form
    researcher = request.form.get('researcher')
    email = request.form.get('email')
    experience = request.form.get('title')

    title = request.form.get('title')
    summary = request.form.get('summary')
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')
    field = request.form.get('field')
    methods = request.form.getlist('methods')
    tools = request.form.getlist('tools')
    years_of_experience = request.form.get("years_of_experience")
    # Generate embedding
    doc_text = f"{title} {summary} Methods: {methods} Tools: {tools}"
    doc_embedding = generate_embedding(doc_text)

    file_paths = []
    
    # Handle missing or empty values
    if years_of_experience is None or years_of_experience.strip() == "":
        years_of_experience = 0
    else:
        years_of_experience = int(years_of_experience)
    if not title or not summary:
        return jsonify({"message": "Missing required fields"}), 400


    # Metadata files
    experience_folder = os.path.join(UPLOAD_FOLDER, secure_filename(experience))
    os.makedirs(experience_folder, exist_ok=True)

    # Save uploaded files
    files = request.files.getlist('files')
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(experience_folder, filename)
            file.save(filepath)
            # Store file metadata
            file_paths.append({
                'filename': filename,
                'path': filepath,
                'upload_date': datetime.now().strftime('%Y-%m-%d')
            })

    # Insert data into MongoDB
    if researcher and email and experience:
        researcher_data = {
            "researcher": researcher,
            "email": email,
            "title": title,
            "summary": summary,
            "start_date": start_date,
            "end_date": end_date,
            "embedding": doc_embedding,

            "attributes": {
                "field": field,
                "methods": methods,
                "tools": tools,
                "years_of_experience": years_of_experience,
                'files': file_paths
            },
        }
        
        # Insert the researcher data into the 'experiences' collection
        collection.insert_one(researcher_data)
        return render_template("result.html", researcher_data=researcher_data)
    else:
        return "All fields are required!", 400


@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        query = request.form['query']
        # Search MongoDB by title, summary, methods, or tools
        experiences = collection.find({
            "$or": [
                {"title": {"$regex": query, "$options": "i"}},
                {"summary": {"$regex": query, "$options": "i"}},
                {"attributes.methods": {"$elemMatch": {"$regex": query, "$options": "i"}}},
                {"attributes.tools": {"$elemMatch": {"$regex": query, "$options": "i"}}}
            ]
        })


        return render_template('results.html', experiences=experiences, query=query)
    return render_template('search.html')
@app.route('/experience/<experience_id>')
def experience_detail(experience_id):
    try:
        # Convert the string experience_id to ObjectId
        experience = collection.find_one({"_id": ObjectId(experience_id)})
        
    except Exception as e:
        # If the ObjectId conversion fails, or no experience is found, abort with 404
        abort(404)
    
    if experience is None:
        abort(404)  # If no experience is found with this ID, abort with a 404 error
    # Process file paths: replace backslashes with forward slashes
    
                 
    return render_template('experience_detail.html', experience=experience)

@app.route('/form')
def form():
    return render_template('form.html')

@app.route("/semanticsearch", methods=["GET", "POST"])
def semanticsearch():
    if request.method == "POST":
        # Get the search query from the form
        query = request.form.get("query")

        # Perform semantic search on the query
        results = executesemanticsearch(query)

        # Return the results to the HTML template
        return render_template("semanticsearch.html", query=query, results=results)
    
    return render_template("semanticsearch.html")
@app.route('/send-email', methods=['GET', 'POST'])
def send_email():
    if 'email' not in session:
        flash("You must be logged in to send an email!", "danger")
        return redirect(url_for('login'))

    if request.method == 'POST':
        to_emails = request.form['to_email'].split(',')  # Get multiple recipients
        cc_emails = request.form.get('cc_email', '').split(',')  # Get CC recipients
        subject = request.form['subject']
        body = request.form['body']

        # Include session email in the message
        full_body = f"Sent by: {session['email']}\n\n{body}"

        msg = Message(subject=subject, recipients=to_emails, cc=cc_emails)
        msg.body = full_body

        # Attach multiple files
        files = request.files.getlist('file')  # Get all uploaded files
        for file in files:
            if file.filename:  # Check if file exists
                msg.attach(file.filename, file.content_type, file.read())

        try:
            mail.send(msg)
            flash(f"Email sent successfully by {session['email']}!", "success")
        except Exception as e:
            flash(f"Failed to send email: {str(e)}", "danger")

        return redirect(url_for('send_email'))

    return render_template('send_email_form.html')

if __name__ == '__main__':
    app.run(debug=True, port=5004)
