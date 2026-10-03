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
    