from flask import Flask, request, jsonify, send_from_directory
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

app = Flask(__name__, static_folder='static')
CORS(app)

# Database credentials from environment variables
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# Configuration
from config import DB_CONFIGS

# --- REMOVED GLOBAL VARIABLES FOR DB_CONNECTION, EMBEDDING_MODEL, INDEX, KEYS, CURRENT_DATABASE ---
# These will now be loaded per request based on the db_name passed in the request.

def get_db_connection(db_name):
    """
    Returns a new database connection object for the given database name.
    Each call to this function will create a new connection.
    Connections should be closed in the finally block of the route.
    """
    config = DB_CONFIGS[db_name]
    try:
        connection = pymysql.connect(
            host=config.get("host", DB_HOST),
            user=config.get("user", DB_USER),
            password=config.get("password", DB_PASSWORD),
            database=config["database"],
        )
        return connection
    except pymysql.Error as e:
        raise Exception(f"Error connecting to database '{db_name}': {e}")


_model_cache = {}
_index_cache = {}
_keys_cache = {}

def load_model_and_index(db_name):
    """
    Loads the Sentence Transformer model and FAISS index for the given database,
    using a cache to avoid reloading for the same database.
    """
    if db_name in _model_cache and db_name in _index_cache and db_name in _keys_cache:
        return _model_cache[db_name], _index_cache[db_name], _keys_cache[db_name]

    config = DB_CONFIGS[db_name]
    
    embedding_model = SentenceTransformer(config["embedding_model_name"])
    index = faiss.read_index(config["index_file"])
    with open(config["mapping_file"], "rb") as f:
        keys = pickle.load(f)

    _model_cache[db_name] = embedding_model
    _index_cache[db_name] = index
    _keys_cache[db_name] = keys

    return embedding_model, index, keys


def get_query(result_key, c=0):
    """Constructs a SQL query to retrieve data based on the result key."""
    tb_c = ["iusnid", "timeperiod_nid", "area_nid", "indicator_nid", "unit_nid", "subgroup_val_nid"]
    result = result_key.split()
    
    # Make sure we have the right number of values
    if len(result) != len(tb_c):
        raise ValueError(f"Expected {len(tb_c)} values in result_key, got {len(result)}")
    
    # Fix spacing in the query for the first case with the 'WHERE' clause
    tb_query = "SELECT data_value FROM ut_data WHERE " if c == 1 else "SELECT data_value FROM data_view WHERE "
    
    conditions = []
    for i, col in enumerate(tb_c):
        # Special handling for empty or 'None' values
        if not result[i] or result[i].lower() == 'none':
            conditions.append(f"{col} IS NULL")
            continue
            
        # Handle the value properly whether it's an integer or float
        try:
            # Check if the value is a float with decimal part
            if '.' in result[i]:
                val = float(result[i])
                # For MySQL, we should cast to DECIMAL for exact representation
                conditions.append(f"{col} = CAST({val} AS DECIMAL(65,30))")
            else:
                # Integer value case - parse to ensure it's valid
                val = int(result[i])
                conditions.append(f"{col} = {val}")
        except ValueError:
            # In case it's not a valid number, try to handle as string
            # Using string replacement before formatting to avoid f-string backslash issues
            safe_val = result[i].replace("'", "''")  # Escape single quotes for SQL
            conditions.append(f"{col} = '{safe_val}'")
    
    tb_query += " AND ".join(conditions)
    return tb_query


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

# Serve the frontend files
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('.', path)

@app.route("/ask", methods=["POST"])
def ask():
    """Handles user queries by selecting the database, loading resources, and performing the search."""
    data = request.json
    user_input = data.get("message")
    db_name = data.get("database")
    use_api = data.get("use_api", False)

    if not user_input:
        return jsonify({"error": "No query provided"}), 400
    if not db_name:
        return jsonify({"error": "No database specified"}), 400
    if db_name not in DB_CONFIGS:
        return jsonify({"error": f"Invalid database name: {db_name}"}), 400

    connection = None
    cursor = None
    try:
        connection = get_db_connection(db_name)
        cursor = connection.cursor()
        embedding_model, index, keys = load_model_and_index(db_name)

        query_embedding = embedding_model.encode([user_input]).astype("float32")
        D, I = index.search(query_embedding, k=3)

        if D[0][0] > 0.59:
            return jsonify({"answer": "Nothing close found", "confidence": D[0].tolist()})

        result_key = keys[I[0][0]]
        
        # result NID for checking and also in case of rollback to rpev ut_data version
        print(f"Result key: {result_key}")

        answer = None
        tb_query = ""
        try:
            tb_query = get_query(result_key)
            print(f"\nExecuting query: {tb_query}")
            print("\n")
            cursor.execute(tb_query)
            answer_row = cursor.fetchone()

            if answer_row and answer_row[0] is not None:
                answer = answer_row[0]
            else:
                tb_query = get_query(result_key, 1)
                print(f"Trying alternate query: {tb_query}")
                cursor.execute(tb_query)
                answer_row = cursor.fetchone()
                if answer_row and answer_row[0] is not None:
                    answer = answer_row[0]
        except Exception as e:
            print(f"Query error: {str(e)}")
            print(f"Failed query: {tb_query}")
            print(f"Result key: {result_key}")
            return jsonify({
                "error": f"Database query error: {str(e)}",
                "matched_key": result_key,
                "sql_query": tb_query if 'tb_query' in locals() else "Query not built",
                "confidence": D[0].tolist(),
                "database": db_name
            }), 500

        connection.commit()

        if answer is None:
            return jsonify({
                "answer": "No data found for matched key", 
                "matched_key": result_key,
                "sql_query": tb_query if 'tb_query' in locals() else "Unknown",
                "confidence": D[0].tolist(),
                "database": db_name
            })

        if use_api:
            sentence = generate_sentence_from_prompt_and_answer(user_input, answer)
            if sentence:
                return jsonify({"generated_sentence": sentence})
            else:
                return jsonify({"error": "Gemini failed to generate a response."}), 500

        return jsonify({
            "answer": answer,
            "query": user_input,
            "matched_key": result_key,
            "sql_query": tb_query,
            "confidence": D[0].tolist(),
            "database": db_name
        })
    except Exception as e:
        print(f"General error: {str(e)}")
        if connection:
            connection.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

if __name__ == "__main__":
    # Use environment variable for port with a default of 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)