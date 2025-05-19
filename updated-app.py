from flask import Flask, request, jsonify
import mysql.connector as pymysql
from sentence_transformers import SentenceTransformer
import pickle
import faiss
from flask_cors import CORS
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Gemini integration
try:
    import google.generativeai as genai
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-2.0-flash")
    else:
        print("Warning: GEMINI_API_KEY not found in environment variables")
        gemini_model = None
except ImportError:
    gemini_model = None

app = Flask(__name__)
CORS(app)

# Database credentials from environment variables
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# Configuration
DB_CONFIGS = {
    "demo-dw-24-sample_main": {
        "host": DB_HOST,
        "user": DB_USER,
        "password": DB_PASSWORD,
        "database": "demo-dw-24-sample_main",
        "embedding_model_name": "all-MiniLM-L6-v2",
        "index_file": "embeddings-24-sample_main.faiss",
        "mapping_file": "index_mapping-24-sample_main.pkl",
    },
    "demo-dw-22-openemis": {
        "host": DB_HOST,
        "user": DB_USER,
        "password": DB_PASSWORD,
        "database": "demo-dw-22-openemis",
        "embedding_model_name": "all-MiniLM-L6-v2",
        "index_file": "embeddings-22-openemis.faiss",
        "mapping_file": "index_mapping-22-openemis.pkl",
    },
    # Add more database configurations as needed
}

# Global variables
db_connection = None
embedding_model = None
index = None
keys = None
current_database = None  # Keep track of the currently selected database


def get_db_connection(db_name):
    """
    Returns a database connection object for the given database name.
    If a connection already exists for the same database, it is reused.
    """
    global db_connection, current_database

    if db_connection and current_database == db_name:
        return db_connection

    if db_connection:
        db_connection.close()  # Close previous connection

    config = DB_CONFIGS[db_name]
    try:
        db_connection = pymysql.connect(
            host=config["host"],
            user=config["user"],
            password=config["password"],
            database=config["database"],
        )
        current_database = db_name  # Update the current database
        return db_connection
    except pymysql.Error as e:
        raise Exception(f"Error connecting to database '{db_name}': {e}")


def load_model_and_index(db_name):
    """Loads the Sentence Transformer model and FAISS index for the given database."""
    global embedding_model, index, keys

    # Check if the model and index are already loaded for the current database
    if embedding_model and index and keys and current_database == db_name:
        return embedding_model, index, keys

    config = DB_CONFIGS[db_name]
    embedding_model = SentenceTransformer(config["embedding_model_name"])
    index = faiss.read_index(config["index_file"])
    with open(config["mapping_file"], "rb") as f:
        keys = pickle.load(f)
    return embedding_model, index, keys


def get_query(result_key):
    """Constructs a SQL query to retrieve data based on the result key."""
    tb_c = ["iusnid", "timeperiod_nid", "area_nid", "indicator_nid", "unit_nid", "subgroup_val_nid"]
    result = result_key.split()
    tb_query = "SELECT data_value FROM ut_data WHERE "
    for i, col in enumerate(tb_c):  # Use enumerate for cleaner loop
        tb_query += f"{col} = {result[i]}"
        if i < len(tb_c) - 1:
            tb_query += " AND "
    return tb_query

def get_view_query(result_key):
    tb_c = ["iusnid", "timeperiod_nid", "area_nid", "indicator_nid", "unit_nid", "subgroup_val_nid"]
    result = result_key.split()
    query = "SELECT * FROM data_view WHERE "
    query += " AND ".join([f"{col} = {result[i]}" for i, col in enumerate(tb_c)])
    return query

def generate_sentence_from_prompt_and_answer(prompt, answer):
    """
    Generates a natural sentence using Gemini given the user's original prompt and the raw answer.
    """
    if gemini_model is None:
        return None

    prompt_text = (
        f"The user asked: \"{prompt}\"\n"
        f"The answer from the database is: \"{answer}\"\n\n"
        f"Using this information, generate a clear and helpful sentence summarizing the response."
    )

    try:
        response = gemini_model.generate_content(prompt_text)

        # Extract main response text
        if hasattr(response, "text") and response.text:
            return response.text.strip()

        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                return candidate.content.parts[0].text.strip()

        return None
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return None


@app.route("/ask", methods=["POST"])
def ask():
    """Handles user queries by selecting the database, loading resources, and performing the search."""
    data = request.json
    user_input = data.get("message")
    db_name = data.get("database")
    use_api = data.get("use_api", False) #Get the value of use_api

    if not user_input:
        return jsonify({"error": "No query provided"}), 400
    if not db_name:
        return jsonify({"error": "No database specified"}), 400

    if db_name not in DB_CONFIGS:
        return jsonify({"error": f"Invalid database name: {db_name}"}), 400

    try:
        connection = get_db_connection(db_name)  # get connection
        cursor = connection.cursor()  # create cursor

        embedding_model, index, keys = load_model_and_index(db_name)  # Load model and index if needed

        query_embedding = embedding_model.encode([user_input]).astype("float32")
        D, I = index.search(query_embedding, k=3)

        if all(d > 0.56 for d in D[0]):
            return jsonify({"answer": "Nothing close found", "confidence": D[0].tolist()})

        result_key = keys[I[0][0]]
        tb_query = get_query(result_key)  # Get the SQL query
        cursor.execute(tb_query)
        answer = cursor.fetchone()[0]
        connection.commit()  # Commit the transaction

        if use_api:
            sentence = generate_sentence_from_prompt_and_answer(user_input, answer)
            if sentence:
                return jsonify({
                    "generated_sentence": sentence
                })
            else:
                return jsonify({"error": "Gemini failed to generate a response."}), 500

        return jsonify({
            "answer": answer,
            "query": user_input,
            "matched_key": result_key,
            "sql_query": tb_query,  # Return the SQL query in the response
            "confidence": D[0].tolist(),
            "database": db_name
        })
    except Exception as e:
        if connection:
            connection.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        if cursor:
            cursor.close()
        # Do NOT close the connection here. It is managed by get_db_connection
        # if connection:
        #     connection.close()


if __name__ == "__main__":
    app.run(debug=True)