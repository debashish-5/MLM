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

    def search_vector(self,query_vector,top_k = 10): 
        if self.vector_index is None or not self.records: 
            return []
        k = min(top_k, len(self.records))
        scores, indices = self.vector_index.search(
            query_vector.astype("float32"), k 
        )
        results = []
        for score, idx in zip(scores[0], indices[0]): 
            if idx < 0:
                continue
            results.append(
                {
                    "id":self.records[int(idx)]['id'], 
                    'score':float(score)
                }
            )
        return results

    def search_keyword(self,query_text, top_k = 10):
        if not self.bm25 or not self.records: 
            return []
        tokens = tokenize(query_text)
        scores = self.bm25
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {
                "id":self.records[int(idx)]['id'], 
                'score':float(scores[int(idx)])
            }
            for idx in ranked_indices if scores[int(idx)] > 0 
        ]

    @staticmethod
    def reciprocal_rank_fusion(result_lists, weights=None, k=60):
        """
        result_lists: ordered lists of {id, score} records.
        RRF uses rank, not the raw retrieval score.
        """
        if weights is None:
            weights = [1.0] * len(result_lists)

        fused_scores = defaultdict(float)

        for weight, results in zip(weights, result_lists):
            seen = set()

            for rank, result in enumerate(results, start=1):
                record_id = result["id"]

                # Prevent duplicate contribution within one list.
                if record_id in seen:
                    continue
                seen.add(record_id)

                fused_scores[record_id] += weight / (k + rank)

        return sorted(
            [
                {"id": record_id, "fusion_score": score}
                for record_id, score in fused_scores.items()
            ],
            key=lambda x: x["fusion_score"],
            reverse=True
        )

    def search(self, text=None, image=None, top_k=5):
        """
        Text-only, image-only, or text + image retrieval.
        """
        if not text and image is None:
            raise ValueError("Provide text, image, or both")

        result_lists = []
        weights = []

        if text and text.strip():
            result_lists.append(
                self.search_keyword(text, top_k=20)
            )
            weights.append(1.0)

            result_lists.append(
                self.search_vector(self.encode_text(text), top_k=20)
            )
            weights.append(1.0)

        if image is not None:
            result_lists.append(
                self.search_vector(self.encode_image(image), top_k=20)
            )
            weights.append(1.0)

        fused = self.reciprocal_rank_fusion(
            result_lists,
            weights=weights
        )

        lookup = {r["id"]: r for r in self.records}

        output = []
        for item in fused[:top_k]:
            record = lookup[item["id"]]

            output.append({
                "record": record,
                "fusion_score": item["fusion_score"]
            })

        return output
