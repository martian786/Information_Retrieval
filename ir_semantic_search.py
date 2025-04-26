"""
Zuveir Jameer
1 April 2025
Retrieval based on sentence embedding only
"""

from elasticsearch import Elasticsearch, helpers
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from pprint import pprint
from credentials import USERNAME, PASSWORD, CERT_FINGERPRINT
import time
import math
from typing import Optional

# FastAPI imports for the front end
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
import uvicorn

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embedding(text):
    """
    Compute the embedding for the given text.
    Returns the embedding as a list of floats.
    """
    encoded_input = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    with torch.no_grad():
        model_output = model(**encoded_input)
    embedding_temp = mean_pooling(model_output, encoded_input['attention_mask'])
    embedding = F.normalize(embedding_temp, p=2, dim=1)
    return embedding[0].cpu().numpy().tolist()

# Connect to Elasticsearch
es = Elasticsearch("https://localhost:9200", basic_auth=(USERNAME, PASSWORD), ssl_assert_fingerprint=CERT_FINGERPRINT)

resultsPerPage = 10

def semantic_search(query_text, page=1, results_per_page=10):
    """
    Search the index using a script_score with 'detailed_descrption_vector_1'
    and our query vector, applying pagination.
    """
    query_vector = get_embedding(query_text)
    offset = (page - 1) * results_per_page
    vector_query_body = {
        "from": offset,
        "size": results_per_page,
        "query": {
            "script_score": {
                "query": {
                    "bool": {
                        "must": [
                            {"exists": {"field": "detailed_descrption_vector_1"}}
                        ]
                    }
                },
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'detailed_descrption_vector_1') + 1.0",
                    "params": {"query_vector": query_vector}
                }
            }
        }
    }
    
    response = es.search(index="ir_dev_sentence_index", body=vector_query_body)
    # Determine total hits (works for ES7+)
    total_hits = response["hits"]["total"]["value"] if isinstance(response["hits"]["total"], dict) else response["hits"]["total"]
    hits_returned = len(response["hits"]["hits"])
    print(f"Total matching documents: {total_hits}, hits returned: {hits_returned}")
    
    results = []
    for hit in response["hits"]["hits"]:
        source = hit["_source"]
        results.append({
            "doc_id": source.get("doc_id", "N/A"),
            "title": source.get("title", "No Title"),
            "summary": source.get("summary", "No Summary"),
            "description": source.get("detailed_descrption", "No Description"),
        })
    return results, total_hits

def get_document(doc_id: str):
    try:
        query_body = {
            "_source": ["doc_id", "title", "summary", "detailed_descrption"],
            "query": {
                "term": {
                    "doc_id.keyword": doc_id
                }
            }
        }
        response = es.search(index="ir_dev_sentence_index", body=query_body)
        hits = response["hits"]["hits"]
        if hits:
            return hits[0]["_source"]
        else:
            return None
    except Exception as e:
        print(f"Error retrieving document {doc_id}: {e}")
        return None

# FastAPI App for the front end
app = FastAPI()

# Home page (using GET so pagination URLs work naturally)
@app.get("/", response_class=HTMLResponse)
async def home():
    html_content = """
    <html>
      <head>
        <title>Zuveir Jameer - Group 6 - Semantic Search Engine</title>
        <style>
          body {
            font-family: Arial, sans-serif;
            # background-color: #f2f2f2;
            background-color: #FFFFFF;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            text-align: center;
          }
          h1 {
            font-size: 3em;
            color: #333;
            margin-bottom: 20px;
          }
          form {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
          }
          input[type="text"] {
            width: 500px;
            padding: 12px;
            font-size: 16px;
            border: 1px solid #ccc;
            border-radius: 5px;
            outline: none;
            box-sizing: border-box;
          }
          input[type="submit"] {
            padding: 12px 20px;
            font-size: 16px;
            border: none;
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            cursor: pointer;
          }
          input[type="submit"]:hover {
            background-color: #45a049;
          }
          /* Move everything slightly higher */
          div {
            margin-top: -250px; /* Adjust as needed */
          }
        </style>
      </head>
      <body>
        <div>
          <h1>Zuveir Jameer - Group 6</h1>
          <p>Semantic Search Engine</p>
          <form action="/search" method="get">
            <input type="text" name="query" placeholder="Enter your query" size="50">
            <input type="image" src="https://img.icons8.com/ios-glyphs/30/000000/search--v1.png" alt="Search">
          </form>
        </div>
      </body>
    </html>
    """
    return html_content

@app.get("/search", response_class=HTMLResponse)
async def search(query: Optional[str] = None, page: int = 1):
    # If no query is provided, simply return the home page or do nothing.
    if not query:
        return await home()
    
    start_time = time.time()
    results_per_page = resultsPerPage
    retrieved_docs, total_hits = semantic_search(query, page=page, results_per_page=results_per_page)
    total_time = time.time() - start_time
    total_pages = math.ceil(total_hits / results_per_page)

    # (Optional) Restrict overall pages to 10. There is no need to display more results.
    total_pages = min(total_pages, 10)

    results_html = f"""
    <html>
    <head>
        <title>Search Results</title>
        <style>
          body {{
            font-family: Arial, sans-serif;
            # background-color: #f9f9f9;
            background-color: #FFFFFF;
            margin: 0;
            padding: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
            padding-top: 20px;
          }}
          .search-box {{
            width: 100%;
            text-align: center;
            margin-bottom: 5px;
          }}
          .search-box form {{
            display: inline-block;
          }}
          .search-box input[type="text"] {{
            width: 600px;
            padding: 12px;
            font-size: 16px;
            border: 1px solid #ccc;
            border-radius: 5px;
            outline: none;
            box-sizing: border-box;
          }}
          .search-box input[type="submit"] {{
            padding: 12px 20px;
            font-size: 16px;
            border: none;
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            cursor: pointer;
          }}
          .search-box input[type="submit"]:hover {{
            background-color: #45a049;
          }}
          h1 {{
            font-size: 2.5em;
            color: #333;
            margin-bottom: 20px;
            text-align: center;
          }}
          p {{
            font-size: 1.1em;
            color: #555;
            margin: 10px 0;
            text-align: center;
          }}
          .results-info {{
            font-size: 1em;
            width: 60%;
            color: #777;
            margin: 10px 0;
            text-align: center;
          }}
          ul {{
            list-style-type: none;
            padding: 0;
            width: 80%;
            max-width: 900px;
            margin: 20px auto;
          }}
          li {{
            background-color: #fff;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
          }}
          a {{
            text-decoration: none;
            color: #1a0dab;
            font-size: 1.3em;
          }}
          a:hover {{
            text-decoration: underline;
          }}
          .back {{
            display: block;
            text-align: center;
            margin-top: 20px;
            font-size: 1.1em;
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            text-decoration: none;
          }}
          .back:hover {{
            background-color: #45a049;
          }}
          .pagination {{
            text-align: center;
            margin: 20px auto 50px; /* 20px top, auto sides, 50px bottom */
            font-size: 1.1em;
            width: 80%;
            max-width: 900px;
          }}
          .pagination a {{
            margin: 0 10px;
            text-decoration: none;
            color: #4CAF50;
            font-weight: bold;
          }}
          .pagination a.disabled {{
            color: #aaa;
            pointer-events: none;
          }}
          .pagination strong {{
            margin: 0 10px;
          }}
        </style>
    </head>
    <body>
        <!-- Search Box at the top -->
        <div class="search-box">
          <form action="/search" method="get">
              <input type="text" name="query" placeholder="Enter your query" value="{query}">
            <input type="image" src="https://img.icons8.com/ios-glyphs/30/000000/search--v1.png" alt="Search" style="vertical-align: middle; margin-left: 5px;">
        </form>

        </div>
        <!--h1>Results</h1-->
        <!--<p class="results-info">Results for: "{query}"</p>-->
        <!--<p class="results-info">Time taken: {total_time:.2f} seconds</p>-->
        <ul>
    """
    for doc in retrieved_docs:
        results_html += (
            f'<li>'
            f'<a href="/doc/{doc["doc_id"]}"><strong>{doc["title"]}</strong></a><br>'
            f'{doc["summary"]}'
            f'</li>'
        )
    results_html += "</ul>"
    
    # Build pagination block: show "previous" link, then page numbers, then "next" link.
    results_html += '<div class="pagination">'
    if page > 1:
        results_html += f'<a href="/search?query={query}&page={page-1}">previous</a>'
    else:
        results_html += f'<a class="disabled">previous</a>'
    
    # Display page numbers (e.g., pages 1 to 10 or up to total_pages if less)
    pages_to_show = min(total_pages, 10)
    for i in range(1, pages_to_show + 1):
        if i == page:
            results_html += f'<strong>{i}</strong>'
        else:
            results_html += f'<a href="/search?query={query}&page={i}">{i}</a>'
    
    if page < total_pages:
        results_html += f'<a href="/search?query={query}&page={page+1}">next</a>'
    else:
        results_html += f'<a class="disabled">next</a>'
    results_html += "</div>"
    
    # results_html += '<p><a href="/" class="back">Back to Search</a></p>'
    results_html += "</body></html>"
    return results_html


@app.get("/doc/{doc_id}", response_class=HTMLResponse)
async def read_document(doc_id: str):
    doc = get_document(doc_id)
    if not doc:
        return HTMLResponse(content=f"<html><body><h1>Document not found</h1><p><a href='/'>Back to Search</a></p></body></html>", status_code=404)
    
    title = doc.get("title", "No Title")
    summary = doc.get("summary", "No Summary")
    description = doc.get("detailed_descrption", "No Description")
    
    html = f"""
    <html>
      <head>
        <title>{title}</title>
        <style>
          body {{
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
            # background-color: #f9f9f9;
            background-color: #FFFFFF;
            display: flex;
            justify-content: center;
            align-items: top;
            height: 100vh;
          }}
          .container {{
            width: 240mm;
            min-height: 600mm;
            padding: 15mm;
            background-color: #fff;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            box-sizing: border-box;
          }}
          h1 {{
            margin-bottom: 20px;
          }}
          .field {{
            margin-bottom: 20px;
          }}
          .field-label {{
            font-weight: bold;
            margin-bottom: 5px;
          }}
          .field-content {{
            margin-left: 10px;
          }}
          a {{
            text-decoration: none;
            color: #007BFF;
          }}
          a:hover {{
            text-decoration: underline;
          }}
        </style>
      </head>
      <body>
        <div class="container">
          <h1>{title}</h1>
          <div class="field">
            <div class="field-label">Summary:</div>
            <div class="field-content">{summary}</div>
          </div>
          <div class="field">
            <div class="field-label">Description:</div>
            <div class="field-content">{description}</div>
          </div>
          <p><a href="/">Back to Search</a></p>
        </div>
      </body>
    </html>
    """
    return html

if __name__ == "__main__":
    # Run the app with: uvicorn ir_main:app --reload
    uvicorn.run("ir_semantic_search:app", host="0.0.0.0", port=8010, reload=True)
