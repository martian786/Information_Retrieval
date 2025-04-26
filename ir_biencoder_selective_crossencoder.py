"""
Zuveir Jameer
1 April 2025
Using Bi-Encoder and Selective Cross-Encoder
"""

from elasticsearch import Elasticsearch, helpers
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
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

# Load bi-encoder model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

# Load cross-encoder model and tokenizer
cross_tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L6-v2")
cross_model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L6-v2")

model.eval()
cross_model.eval()
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
cross_model.to(device)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # All token embeddings
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
    # Move inputs to the same device as the model
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    with torch.no_grad():
        model_output = model(**encoded_input)
    embedding_temp = mean_pooling(model_output, encoded_input['attention_mask'])
    embedding = F.normalize(embedding_temp, p=2, dim=1)
    return embedding[0].cpu().numpy().tolist()

# Connect to Elasticsearch
es = Elasticsearch("https://localhost:9200", basic_auth=(USERNAME, PASSWORD), ssl_assert_fingerprint=CERT_FINGERPRINT)

# ------------------ Dense Retrieval using Bi-Encoder ------------------
def search_detailed_descrption(query_vector, K=100):
    """
    Retrieve candidate documents using dense vector search with cosine similarity.
    Assumes each document in ES has a pre-computed vector in the field 
    'detailed_descrption_vector_1'.
    Returns a list of tuples: (doc_id, similarity_score)
    """
    search_body = {
      "query": {
        "script_score": {
          "query": {
            "bool": {
              "must": [
                {"match_all": {}},
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
    response = es.search(index="ir_dev_sentence_index", body=search_body, size=K)
    results = []
    for hit in response['hits']['hits']:
        doc_id = hit["_source"]["doc_id"]
        score = hit["_score"]
        results.append((doc_id, score))
    return results

# ------------------ Cross-Encoder Re-Ranking ------------------
def get_document_text(doc_id):
    """
    Retrieve the detailed description for a document given its ID.
    """
    query = {
        "query": {
            "term": {"doc_id.keyword": doc_id}
        }
    }
    response = es.search(index="ir_dev_sentence_index", body=query, size=1)
    hits = response.get("hits", {}).get("hits", [])
    if hits:
        doc_text = hits[0]["_source"].get("detailed_descrption", "").strip()
        return doc_text if doc_text else None
    else:
        return None

def cross_encoder_rerank(query_text, candidate_doc_ids, top_k=100):
    candidate_pairs = []
    valid_doc_ids = []
    for doc_id in candidate_doc_ids:
        try:
            doc_text = get_document_text(doc_id)
            if doc_text is None or not doc_text.strip():
                continue
            candidate_pairs.append((query_text, doc_text))
            valid_doc_ids.append(doc_id)
        except Exception as e:
            print(f"Error fetching doc {doc_id}: {e}")
            continue

    if not candidate_pairs:
        print("No valid candidate pairs found; returning original candidates.")
        return candidate_doc_ids

    inputs = cross_tokenizer.batch_encode_plus(
        candidate_pairs, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    inputs = {k: v.to(cross_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = cross_model(**inputs)

    # Handle both regression and classification outputs
    if outputs.logits.shape[-1] == 1:
        scores = outputs.logits.squeeze(-1)
    else:
        scores = torch.softmax(outputs.logits, dim=1)[:, 1]

    ranked = sorted(
        zip(valid_doc_ids, scores.tolist()), 
        key=lambda x: x[1], 
        reverse=True
    )

    if top_k is not None:
        ranked = ranked[:top_k]

    return [doc_id for doc_id, score in ranked]

# ------------------ Document Retrieval ------------------
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

# ------------------ FastAPI App ------------------
app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def home():
    html_content = """
    <html>
      <head>
        <title>Zuveir Jameer - Group 6 - Bi-encoder & Selective Cross-encoder Search Engine</title>
        <style>
          body {
            font-family: Arial, sans-serif;
            background-color: #ffffcc;
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
          input[type="image"] {
            vertical-align: middle;
            margin-left: 5px;
            cursor: pointer;
          }
          div {
            margin-top: -250px;
          }
        </style>
      </head>
      <body>
        <div>
          <h1>Zuveir Jameer - Group 6</h1>
          <p>Bi-encoder & Selective Cross-encoder Search Engine</p>
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
    # Stage 1: Dense retrieval using bi-encoder (get up to K candidates)
    query_vector = get_embedding(query)
    retrieved_docs_with_scores = search_detailed_descrption(query_vector, K=100)
    if not retrieved_docs_with_scores:
        return HTMLResponse(content=f"<html><body><h1>No results found for query: {query}</h1></body></html>")
    # Separate candidate doc_ids and scores
    retrieved_docs = [doc for doc, score in retrieved_docs_with_scores]
 
    top_bi_score = retrieved_docs_with_scores[0][1]

    # Define a threshold on the score range (tune this value as needed)
    similarity_threshold = 1.6 
    if top_bi_score < similarity_threshold:
        print(f"Top bi-encoder score {top_bi_score:.4f} is below threshold {similarity_threshold}, applying cross-encoder re-ranking")
        try:
            re_ranked_doc_ids = cross_encoder_rerank(query, retrieved_docs, top_k=10)
        except Exception as e:
            print(f"Error in cross-encoder re-ranking: {e}")
            re_ranked_doc_ids = retrieved_docs
    else:
        print(f"Top bi-encoder score {top_bi_score:.4f} exceeds threshold {similarity_threshold}, skipping cross-encoder re-ranking")
        re_ranked_doc_ids = retrieved_docs

    # For pagination: use re-ranked list as final results
    final_results = re_ranked_doc_ids
    total_results = len(final_results)
    results_per_page = 10
    total_pages = math.ceil(total_results / results_per_page)
    start_index = (page - 1) * results_per_page
    end_index = start_index + results_per_page
    page_doc_ids = final_results[start_index:end_index]
    
    # Retrieve full document details for display.
    final_docs = []
    for doc_id in page_doc_ids:
        doc = get_document(doc_id)
        if doc:
            final_docs.append(doc)
    total_time = time.time() - start_time

    # Build HTML response.
    results_html = f"""
    <html>
    <head>
        <title>Search Results - selective cross-encoder</title>
        <style>
          body {{
            font-family: Arial, sans-serif;
            background-color: #ffffcc;
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
            margin-bottom: 20px;
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
          .search-box input[type="image"] {{
            vertical-align: middle;
            margin-left: 5px;
            cursor: pointer;
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
          .pagination {{
            text-align: center;
            margin: 20px auto 50px;
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
        </style>
    </head>
    <body>
        <div class="search-box">
          <form action="/search" method="get">
              <input type="text" name="query" placeholder="Enter your query" value="{query}">
              <input type="image" src="https://img.icons8.com/ios-glyphs/30/000000/search--v1.png" alt="Search">
          </form>
        </div>
        <p class="results-info">Results for: "{query}"</p>
        <p class="results-info">Time taken: {total_time:.2f} seconds</p>
        <ul>
    """
    for doc in final_docs:
        results_html += (
            f'<li>'
            f'<a href="/doc/{doc["doc_id"]}"><strong>{doc["title"]}</strong></a><br>'
            f'{doc["summary"]}'
            f'</li>'
        )
    results_html += "</ul>"
    
    # Pagination block.
    results_html += '<div class="pagination">'
    if page > 1:
        results_html += f'<a href="/search?query={query}&page={page-1}">previous</a>'
    else:
        results_html += '<a class="disabled">previous</a>'
    pages_to_show = min(total_pages, 10)
    for i in range(1, pages_to_show + 1):
        if i == page:
            results_html += f'<strong>{i}</strong>'
        else:
            results_html += f'<a href="/search?query={query}&page={i}">{i}</a>'
    if page < total_pages:
        results_html += f'<a href="/search?query={query}&page={page+1}">next</a>'
    else:
        results_html += '<a class="disabled">next</a>'
    results_html += "</div>"
    
    results_html += '<p><a href="/" class="back">Back to Search</a></p>'
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
            background-color: #ffffcc;
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
    uvicorn.run("ir_biencoder_selective_crossencoder:app", host="0.0.0.0", port=8070, reload=True)
