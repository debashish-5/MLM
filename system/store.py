import json
import uuid
import threading
from pathlib import Path
import faiss
import numpy as np
from system.config import settings

class VectorStore:
    def __init__(self,dimension:str):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.records:list[dict] = []
        self.lock = threading.RLock()
        self._load()

    def _load(self):
        with self.lock:
            if settings.metadata_path.exists():
                self.records = json.loads(
                    settings.metadata_path.read_text(encoding="utf-8")
                )
            if settings.index_path.exists():
                loaded = faiss.read_index(str(settings.index_path))
                if loaded.d != self.dimension:
                    raise ValueError("Save FAISS index dimension mismatch")
                if loaded.ntotal != len(self.records):
                    raise ValueError("FAISS and Metadata record count differ")
                self.index = loaded
            elif self.records:
                vectors = np.array(
                    [r['embedding'] for r in self.records], 
                    dtype="float32"
                )
                self.index.add(vectors)
                self._save()
        def _save(self):
            settings.metadata_path.write_text(
                json.dumps(self.records, ensure_ascii=True, indent=2), 
                encoding="utf-8"
            )
            faiss.write_index(self.index, str(settings.index_path))

        def add(self,record:dict, vector:np.ndarray):
            vector = np.asarray(vector, dtype="float32").reshape(1,-1)
            if vector.shape[1] != self.dimension:
                raise ValueError("Embedding dimension mismatch")
            with self.lock:
                record = {
                    **record, 
                    "id":record.get("id") or str(uuid.uuid4()), 
                    "vector":vector[0].tolist()
                }
                self.index.add(vector)
                self.records.append(record)
                self._save()
                return {k:v for k,v in record.items() if k != "embedding"}


        def search(self, vector:np.ndarray, top_k:int = 5):
            vector = np.asarray(vector, dtype="float32").reshape(-1,1)
            with self.lock:
                if self.index.ntotal == 0:
                    return []
                k = min(top_k, self.index.ntotal)
                scores,indices = self.index.search(vector, k)
                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx < 0:
                        continue
                    record = self.record[int(idx)]
                    results.append({
                        "records":{
                            k:v for  k,v in record.items() if k != "embedding"
                        }, 
                        "score":float(score)
                    })
                return results

        def count(self):
            with self.lock:
                return self.index.ntotal    
        


                
            