import re 
from collections import defaultdict
import faiss 
import numpy as np 
import torch
import open_clip
from PIL import Image
from rank_bm25 import BM25Okapi


def tokenize(text:str) -> list[str]:
    #preserve codes like E102 and XJ-500
    return re.findall(r"[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*", text.lower())

class HybridRetriever: 
    def __init__(self,model_name = "ViT-B-32", pretrained = "laion2b_s34b_b79k"): 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = (
            open_clip.create_model_and_transforms(
                model_name,pretrained=pretrained,device=self.device
            )
        )
        self.tokenizer = open_clip.get_tokenizer()
        self.model.eval()
        self.records = []
        self.vector_index = None
        self.bm25 = None    
        self.dimension = None

    @torch.inference_mode()
    def encode_text(self, text): 
        tokens = self.tokenizer([text]).to(self.device)
        vector = self.model.encode_text(tokens).float()
        vector = vector / vector.norm(
            dim=-1, 
            keepdim=True
        ).clamp(min=1e-12)
        return vector.cpu().numpy().astype('float32')

    @torch.inference_mode()
    def encode_images(self,image): 
        tensor = self.preprocess(
            image.convert('RGB')
        ).unsqueeze(0).to(self.device)
        vector = self.model.encode_image(tensor).float()
        vector = vector / vector.norm(
            dim = -1, 
            keepdim = True
        ).clamp(min=1e-12)
        return vector.cpu().numpy().astype('float32')


    def add_text(self,record): 
        """
        Index text for both lexical and semantic retrieval.
        `record` must include id, title, text, and source.
        """
        if any(r['id'] == record['id'] for r in self.records): 
            raise ValueError("Duplicate record ID:",self.record['id'])

        record = {
            **record, 
            "modality":"text",
            "search_text":(
                f"{record.get("title",'')}"
                f"{record.get("title",'')}"
            )
        }
        vector = self.model.encode_text(record['search_text'])
        self._add_record(record, vector)

    def add_image(self,record, image): 
        """
        Index image semantics in the CLIP vector space.
        Also make caption and OCR text available to BM25.
        """
        if any(r['id'] == record['id'] for r in self.records): 
            raise ValueError(f"Duplicate record ID:{self.record['id']}")

        record = {
            **record, 
            "modality":"image", 
            "search_text":" ".join([record.get("title",""), record.get("caption",""), record.get('ocr_text',"")])
        }

        vector = self.model.encode_image(image)
        self._add_record(record,vector)

    def _add_record(self,record, vector): 
        vector = np.asarray(vector,dtype="float32").reshape(1,-1)
        
        if self.dimension is None: 
            self.dimension = vector.shape[1]
            self.vector_index = faiss.IndexFlatIP(self.dimension)  
        if vector.shape[1] != self.dimension: 
            raise ValueError("Inconsistent vector dimensions")
        self.vector_index.add(vector)
        self.records.append(record)

        # Rebuild BM25 for this small teaching implementation.
        # For larger collections, update a persistent lexical index.
        corpus = [
            tokenize(r['search_text']) for r in self.records
        ]
        self.bm25 = BM25Okapi(corpus=corpus)