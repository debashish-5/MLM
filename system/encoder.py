import torch
import open_clip 
from PIL import Image
from system.config import settings
import clip 


class ClipEncoder:
    def __int__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _ , self.preprocess = open_clip.create_model_and_transformers(
            settings.clip_model, 
            pretrained = settings.clip_pretrained, 
            device = self.device
        )
        self.tokenizer = open_clip.get_tokenizer(settings.clip_model)
        self.model.eval()

    @torch.inference_mode()
    def encode_text(self,text:str):
        tokens = self.tokenizer([text]).to(self.device)
        vectors = self.model.encode_text(tokens)
        vectors = vectors.float()
        vectors = vectors / vectors.norm(dim=-1, keepdim= True).clamp(min=1e-12)
        return vectors[0].cpu().numpy()

    @torch.inference_mode()
    def encode_image(self,image:Image.Image):
        image = image.convert("RGB")
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        vectors = self.model.encode_images(tensor)
        vectors = vectors.float()
        vectors = vectors / vectors.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return vectors[0].cpu().numpy()
    