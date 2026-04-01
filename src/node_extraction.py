import os
import pandas as pd
import numpy as np
import difflib
from typing import List, Dict, Any, Optional

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    SEMANTIC_SEARCH_AVAILABLE = True
except ImportError:
    SEMANTIC_SEARCH_AVAILABLE = False
    print("[WARNING] faiss-cpu or sentence-transformers not found. Falling back to difflib only.")

class NodeExtractor:
    """
    Resolves fuzzy or misspelled entity names (players, clubs, etc.) 
    to their canonical forms using embeddings and indexing.
    """
    
    COLUMNS = ['name', 'position', 'nationality', 'club', 'league']
    
    # Map long-form position names to database short-forms
    POSITION_MAP = {
        "striker": "ST",
        "left wing": "LW",
        "right wing": "RW",
        "centre forward": "CF",
        "left midfield": "LM",
        "right midfield": "RM",
        "centre midfield": "CM",
        "centre attacking midfield": "CAM",
        "centre defensive midfield": "CDM",
        "centre back": "CB",
        "right back": "RB",
        "left back": "LB",
        "right wing back": "RWB",
        "left wing back": "LWB",
        "goal keeper": "GK"
    }

    # Map tool parameter names to CSV column names
    PARAM_MAP = {
        'player_name': 'name',
        'position': 'position',
        'nationality': 'nationality',
        'nation': 'nationality',
        'club': 'club',
        'league': 'league'
    }

    def __init__(self, data_path: str = 'data/player-data-cleaned.csv'):
        self.data_path = data_path
        self.df = None
        self.indices = {}
        self.id_to_value = {}
        self.model = None
        
        if os.path.exists(data_path):
            self.df = pd.read_csv(data_path)
            self._build_indices()
        else:
            print(f"[ERROR] Data file {data_path} not found.")

    def _build_indices(self):
        """Build FAISS indices for semantic search on unique entity values."""
        if not SEMANTIC_SEARCH_AVAILABLE or self.df is None:
            return

        print("[NodeExtractor] Initializing embedding model and building indices...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        for col in self.COLUMNS:
            if col not in self.df.columns:
                continue
                
            unique_values = self.df[col].dropna().unique().tolist()
            if not unique_values:
                continue

            # Special case for position: index both short and long forms
            if col == 'position':
                indexed_values = list(set(unique_values + list(self.POSITION_MAP.keys())))
            else:
                indexed_values = unique_values
                
            # Compute embeddings
            embeddings = self.model.encode(indexed_values, show_progress_bar=False)
            embeddings = np.array(embeddings).astype('float32')
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Index
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings)
            
            self.indices[col] = index
            self.id_to_value[col] = indexed_values
            
        print(f"[NodeExtractor] Successfully indexed {len(self.indices)} categories.")

    def resolve_entity(self, text: str, entity_type: str) -> str:
        """
        Maps a fuzzy input string to the closest canonical value in the database.
        Uses semantic embedding search if available, falls back to fuzzy string matching.
        """
        if not text or not isinstance(text, str):
            return text
            
        col = self.PARAM_MAP.get(entity_type, entity_type)
        if col not in self.COLUMNS:
            return text

        # 1. Position Mapping (Special Case)
        if col == 'position':
            normalized = text.strip().lower()
            # Direct match to short form (case-insensitive check)
            short_forms = [v.lower() for v in self.POSITION_MAP.values()]
            if normalized in short_forms:
                return text.upper()
            
            # Match to long form
            if normalized in self.POSITION_MAP:
                resolved = self.POSITION_MAP[normalized]
                print(f"[NodeExtractor] Position Map: '{text}' -> '{resolved}'")
                return resolved

        # 2. Semantic Search (FAISS)
        if SEMANTIC_SEARCH_AVAILABLE and col in self.indices:
            try:
                query_vec = self.model.encode([text])
                query_vec = np.array(query_vec).astype('float32')
                faiss.normalize_L2(query_vec)
                
                D, I = self.indices[col].search(query_vec, 1)
                best_match = self.id_to_value[col][I[0][0]]
                score = D[0][0]
                
                if score > 0.6:  # Similarity threshold
                    if best_match.lower() != text.lower():
                        print(f"[NodeExtractor] Semantic Resolve: '{text}' -> '{best_match}' (score: {score:.2f})")
                    
                    # Ensure position returns the short form
                    if col == 'position' and best_match.lower() in self.POSITION_MAP:
                        return self.POSITION_MAP[best_match.lower()]
                    
                    return best_match
            except Exception as e:
                print(f"[ERROR] Semantic search failed for '{text}': {e}")

        # 2. Fuzzy Matching (difflib) fallback
        if col in self.id_to_value:
            candidates = self.id_to_value[col]
        elif self.df is not None and col in self.df.columns:
            candidates = self.df[col].dropna().unique().tolist()
            self.id_to_value[col] = candidates
        else:
            return text

        matches = difflib.get_close_matches(text, candidates, n=1, cutoff=0.6)
        if matches:
            best_match = matches[0]
            if best_match.lower() != text.lower():
                print(f"[NodeExtractor] Fuzzy Resolve: '{text}' -> '{best_match}'")
            
            # Ensure position returns the short form
            if col == 'position' and best_match.lower() in self.POSITION_MAP:
                return self.POSITION_MAP[best_match.lower()]
                
            return best_match

        return text

    def resolve_params(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Resolves all eligible parameters for a given tool call."""
        resolved = {}
        for k, v in params.items():
            if isinstance(v, str) and k in self.PARAM_MAP:
                resolved[k] = self.resolve_entity(v, k)
            elif isinstance(v, list) and k == 'player_names':
                # Special handling for lists of players if needed (decoding handled by tool itself)
                resolved[k] = v
            else:
                resolved[k] = v
        return resolved

# Singleton instance
extractor = NodeExtractor()
