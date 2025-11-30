"""
Recipe Recommendation API - Ready for Render Deployment
"""

import pandas as pd
import numpy as np
import re
import json
import os
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ---------- Configuration ----------
DATA_FILE = "recipes.csv"
OUTPUT_JSON = "recipes_output.json"
MAX_ROWS = 50000  # Adjust based on your Render plan

# ---------- Pydantic Models ----------

class RecommendRequest(BaseModel):
    ingredients: List[str]
    top_n: int = 10
    min_match: float = 25.0

class RecipeItem(BaseModel):
    rank: int
    recipe_name: str
    similarity_percent: float
    ingredients_used: int
    total_needed: int
    match_rate: str
    ingredients: str

class RecommendResponse(BaseModel):
    status: str
    count: int
    results: List[RecipeItem]

# ---------- Data Processing Functions ----------

def clean_ingredients(text):
    """Clean and normalize ingredient text"""
    if pd.isna(text) or text == "":
        return ""
    text = str(text).lower()
    text = re.sub(r"[\\\[\]\"']", " ", text)
    text = re.sub(r"[^a-z0-9\s/,.-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_and_preprocess_data():
    """Load CSV and preprocess ingredients"""
    print("Loading data...")
    df = pd.read_csv(DATA_FILE, low_memory=False, nrows=MAX_ROWS, on_bad_lines='skip')
    df.columns = df.columns.str.lower()
    
    # Find column names
    NAME_COL = next((c for c in ["name", "title", "recipe_name"] if c in df.columns), None)
    if NAME_COL is None:
        raise ValueError(f"No suitable name column found. Available: {df.columns.tolist()}")
    
    ING_COL = next((c for c in ["ingredients", "ingredient_list", "recipeingredientparts"] if c in df.columns), None)
    if ING_COL is None:
        raise ValueError(f"No suitable ingredient column found. Available: {df.columns.tolist()}")
    
    print(f"Using columns: {NAME_COL}, {ING_COL}")
    
    # Clean ingredients
    df["ingredients_clean"] = df[ING_COL].astype(str).apply(clean_ingredients)
    df = df[df["ingredients_clean"].str.len() > 10].reset_index(drop=True)
    
    print(f"Recipes after cleaning: {len(df)}")
    
    return df, NAME_COL, ING_COL

def build_features(df):
    """Build TF-IDF, SVD, and BERT features"""
    print("Building TF-IDF features...")
    tfidf = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=15000,
        sublinear_tf=True,
        min_df=3,
        max_df=0.98,
        stop_words="english"
    )
    tfidf_matrix = tfidf.fit_transform(df["ingredients_clean"])
    print(f"TF-IDF shape: {tfidf_matrix.shape}")
    
    print("Building SVD features...")
    svd = TruncatedSVD(n_components=256, random_state=42)
    tfidf_reduced = svd.fit_transform(tfidf_matrix)
    print(f"SVD shape: {tfidf_reduced.shape}")
    
    print("Building BERT embeddings...")
    bert_model = SentenceTransformer("all-MiniLM-L6-v2")
    bert_embeddings = bert_model.encode(df["ingredients_clean"].tolist(), show_progress_bar=True)
    print(f"BERT shape: {bert_embeddings.shape}")
    
    return tfidf_matrix, tfidf_reduced, bert_embeddings, tfidf, svd, bert_model

def save_recipes_json(df, name_col, ing_col):
    """Save recipes to JSON file"""
    print("Saving recipes to JSON...")
    output_data = []
    for _, row in df.iterrows():
        output_data.append({
            "name": str(row[name_col]),
            "ingredients": str(row[ing_col]),
            "ingredients_clean": str(row["ingredients_clean"])
        })
    
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Saved {len(output_data)} recipes to {OUTPUT_JSON}")

# ---------- Recommender Class ----------

class SuperCookRecommender:
    """Main ML model for recipe recommendations"""
    
    def __init__(self, df, tfidf_matrix, tfidf_reduced, bert_embeddings, 
                 tfidf, svd, bert_model, name_col, ing_col):
        self.df = df.reset_index(drop=True)
        self.tfidf_matrix = tfidf_matrix
        self.tfidf_reduced = tfidf_reduced
        self.bert_embeddings = bert_embeddings
        self.tfidf = tfidf
        self.svd = svd
        self.bert_model = bert_model
        self.name_col = name_col
        self.ing_col = ing_col

    @staticmethod
    def _prep_user_text(ingredients_list):
        text = " ".join(ingredients_list)
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s/,.-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _get_user_features(self, ingredients_list):
        user_text = self._prep_user_text(ingredients_list)
        user_tfidf = self.tfidf.transform([user_text])
        user_svd = self.svd.transform(user_tfidf)
        user_bert = self.bert_model.encode([user_text])
        return user_tfidf, user_svd, user_bert

    def _ensemble_similarity(self, user_tfidf, user_svd, user_bert):
        sim_tfidf = cosine_similarity(user_tfidf, self.tfidf_matrix).flatten()
        sim_svd = cosine_similarity(user_svd, self.tfidf_reduced).flatten()
        sim_bert = cosine_similarity(user_bert, self.bert_embeddings).flatten()
        
        # Weighted ensemble
        sim = 0.7 * sim_tfidf + 0.1 * sim_svd + 0.2 * sim_bert
        return sim

    def recommend(self, user_ingredients, top_n=10, min_match=20.0):
        """Generate recipe recommendations"""
        user_tfidf, user_svd, user_bert = self._get_user_features(user_ingredients)
        sims = self._ensemble_similarity(user_tfidf, user_svd, user_bert)

        idx_sorted = np.argsort(sims)[::-1][:top_n]
        results = []

        user_set = set([w.lower() for w in user_ingredients])

        for rank, idx in enumerate(idx_sorted, start=1):
            score_pct = float(sims[idx] * 100.0)
            if score_pct < min_match:
                continue

            ing_tokens = set(self.df.loc[idx, "ingredients_clean"].split())
            common = user_set & ing_tokens

            results.append({
                "rank": rank,
                "recipe_name": self.df.loc[idx, self.name_col],
                "similarity_percent": round(score_pct, 2),
                "ingredients_used": len(common),
                "total_needed": len(ing_tokens),
                "match_rate": f"{len(common)}/{len(ing_tokens)}",
                "ingredients": self.df.loc[idx, self.ing_col],
            })

        return pd.DataFrame(results)

# ---------- Initialize Global Variables ----------

print("Initializing application...")
df, NAME_COL, ING_COL = load_and_preprocess_data()
tfidf_matrix, tfidf_reduced, bert_embeddings, tfidf, svd, bert_model = build_features(df)
save_recipes_json(df, NAME_COL, ING_COL)

recommender = SuperCookRecommender(
    df=df,
    tfidf_matrix=tfidf_matrix,
    tfidf_reduced=tfidf_reduced,
    bert_embeddings=bert_embeddings,
    tfidf=tfidf,
    svd=svd,
    bert_model=bert_model,
    name_col=NAME_COL,
    ing_col=ING_COL
)

print("Recommender initialized successfully!")

# ---------- FastAPI App ----------

app = FastAPI(
    title="Recipe Recommendation API",
    description="ML-powered recipe recommendations based on available ingredients",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your React app domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "Recipe Recommendation API is running",
        "total_recipes": len(df),
        "endpoints": {
            "recipes": "/recipes",
            "recommend": "/recommend",
            "health": "/health"
        }
    }

@app.get("/health")
def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "recipes_loaded": len(df),
        "model_ready": recommender is not None
    }

@app.get("/recipes")
def get_all_recipes():
    """Get all recipes as JSON - Use this endpoint in your React app"""
    try:
        if not os.path.exists(OUTPUT_JSON):
            raise HTTPException(status_code=404, detail="Recipes JSON file not found")
        
        with open(OUTPUT_JSON, 'r') as f:
            recipes_data = json.load(f)
        
        return {
            "status": "success",
            "count": len(recipes_data),
            "recipes": recipes_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading recipes: {str(e)}")

@app.post("/recommend", response_model=RecommendResponse)
def recommend_endpoint(payload: RecommendRequest):
    """
    Get recipe recommendations based on available ingredients
    
    Example request:
    {
        "ingredients": ["chicken", "garlic", "onion"],
        "top_n": 10,
        "min_match": 25.0
    }
    """
    try:
        if not payload.ingredients:
            raise HTTPException(status_code=400, detail="Ingredients list cannot be empty")
        
        # Get recommendations
        df_res = recommender.recommend(
            user_ingredients=payload.ingredients,
            top_n=payload.top_n,
            min_match=payload.min_match,
        )
        
        # Convert to response format
        results = []
        for _, row in df_res.iterrows():
            results.append(
                RecipeItem(
                    rank=int(row["rank"]),
                    recipe_name=str(row["recipe_name"]),
                    similarity_percent=float(row["similarity_percent"]),
                    ingredients_used=int(row["ingredients_used"]),
                    total_needed=int(row["total_needed"]),
                    match_rate=str(row["match_rate"]),
                    ingredients=str(row["ingredients"]),
                )
            )
        
        return RecommendResponse(
            status="success",
            count=len(results),
            results=results,
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")

# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)