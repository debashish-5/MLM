from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name:str = "Multimodal Query Engine"
    data_dir: Path = Path("data")
    image_dir: Path = Path("data/images")
    metadata_path: Path = Path("metadata.json")
    index_path:Path = Path("data/index.faiss")
    clip_model:str = "ViT-B-32"
    clip_pretrained:str = "laion2b_s34b_b79k"
    max_upload_bytes:int = 10 * 1024 * 1024
    max_query_images:int = 4
    default_top_k:int = 5
    max_top_k:int = 5
    class Config:
        env_file= ".env"
        extra = "ignore"

settings = Settings()
settings.data_dir.mkdir(parents=True,exist_ok=True)
settings.image_dir.mkdir(parents=True,exist_ok=True)


    