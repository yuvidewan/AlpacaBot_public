# Database API with Semantic Search

A Flask API that connects to a MySQL database and uses semantic search to find relevant data.

## Features

- Query data using natural language
- Uses FAISS for semantic similarity search
- Supports multiple databases
- Optional integration with Google Gemini for natural language response generation

## Setup

### Prerequisites

- Python 3.8+
- MySQL server
- FAISS index files for your databases
- Python libraries (see requirements.txt)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/database-api.git
   cd database-api
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file based on the example:
   ```
   cp .env.example .env
   ```

4. Edit the `.env` file with your database credentials and API keys:
   ```
   # Database credentials
   DB_HOST=localhost
   DB_USER=your_username
   DB_PASSWORD=your_password
   
   # API keys
   GEMINI_API_KEY=your_gemini_api_key
   ```

### Configuration

The application supports multiple databases. Configure your databases in `config.py`:

```python
DB_CONFIGS = {
    "your-database-name": {
        "database": "your-database-name",
        "embedding_model_name": "all-MiniLM-L6-v2",
        "index_file": "path/to/your/embeddings.faiss",
        "mapping_file": "path/to/your/index_mapping.pkl",
    },
    # Add more database configurations as needed
}
```

## Running the Application

Start the Flask application:

```
python app.py
```

The API will be available at `http://localhost:5000`.

## API Usage

### Ask Endpoint

`POST /ask`

Parameters:
- `message`: The natural language query
- `database`: The database to query
- `use_api` (optional): Whether to use Gemini API for response generation

Example Request:
```json
{
  "message": "What is the population of New York?",
  "database": "demo-dw-24-sample_main",
  "use_api": true
}
```

Example Response:
```json
{
  "answer": "8336817",
  "query": "What is the population of New York?",
  "matched_key": "1 5 24 12 1 0",
  "sql_query": "SELECT data_value FROM ut_data WHERE iusnid = 1 AND timeperiod_nid = 5 AND area_nid = 24 AND indicator_nid = 12 AND unit_nid = 1 AND subgroup_val_nid = 0",
  "confidence": [0.21, 0.35, 0.48],
  "database": "demo-dw-24-sample_main"
}
```

## License

[Your License Here]