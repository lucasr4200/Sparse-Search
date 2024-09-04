from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import psycopg2
from typing import List, Optional
 
app = FastAPI()
 
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
 
#Load model and tokenizer
model_id = 'opensearch-project/opensearch-neural-sparse-encoding-v1'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id)

#db connection
db_params = {
    "host": "ip",
    "database": "database",
    "user": "database",
    "password": "password",
    "port": "port"
}
 
class SearchResult(BaseModel):
    title: str
    description: Optional[str]
    cosineSimilarity: float
 
class SearchResponse(BaseModel):
    isSuccess: bool
    response: List[SearchResult]
    errors: Optional[str]
 
@app.get("/search", response_model=SearchResponse)
async def search(query: str = Query(..., description="Search query")):
    try:
        # Tokenize and get model output
        tokens = tokenizer(
            query, return_tensors='pt',
            padding=True, truncation=True
        )
        output = model(**tokens)

        vecs = torch.max(
            torch.log(1 + torch.relu(output.logits)) * tokens.attention_mask.unsqueeze(-1), dim=1
        )[0].squeeze()
 
        #Extract non-zero positions and values
        cols = vecs.nonzero().squeeze().cpu().tolist()
        weights = vecs[cols].cpu().tolist()
        sparse_dict = dict(zip(cols, weights))
 
        sparse_vector_string = f"'{sparse_dict}/30522'"
 
        #Database query
        db_query = f"""
        SELECT column, 
        1 - (column <=> {sparse_vector_string}) AS cosine_similarity 
        FROM table 
        ORDER BY cosine_similarity DESC 
        LIMIT 10;
        """
 
        #Execute database query
        with psycopg2.connect(**db_params) as connection:
            with connection.cursor() as cursor:
                cursor.execute(db_query)
                results = cursor.fetchall()
 
        #Process results
        response_items = []
        for result in results:
            title, description, similarity = result
            response_item = SearchResult(
                title=title,
                description=description if description else None,
                cosineSimilarity=similarity,
            )
            response_items.append(response_item)
 
        return SearchResponse(isSuccess=True, response=response_items, errors=None)
 
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)