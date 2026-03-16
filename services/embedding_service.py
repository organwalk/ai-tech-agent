from typing import List
import logging
from volcenginesdkarkruntime import Ark
from config import Config

logger = logging.getLogger(__name__)


class EmbeddingService:
    _instance = None
    _client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._client = Ark(
                api_key=Config.API_KEY,
                base_url=Config.BASE_URL,
                timeout=Config.CLIENT_TIMEOUT
            )
        return cls._instance

    def __init__(self):
        self.client = self._client
        self.model_id = Config.EMBED_MODEL_ID

    def get_embedding(self, text: str) -> List[float]:
        try:
            response = self.client.multimodal_embeddings.create(
                model=self.model_id,
                input=[{"type": "text", "text": text}]
            )
            return response.data.embedding
        except Exception as e:
            logger.error("[embedding][error] single request failed: %s", e)
            raise e

    def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        
        for text in texts:
            try:
                payload = [{"type": "text", "text": text}]
                response = self.client.multimodal_embeddings.create(
                    model=self.model_id,
                    input=payload
                )
                
                if response.data and len(response.data) > 0:
                    item = response.data[0]
                    vector = item.embedding
                    
                    if isinstance(vector, list):
                        embeddings.append(vector)
                    else:
                        logger.warning("[embedding][warn] embedding vector type invalid: %s", type(vector))
            
            except Exception as e:
                logger.error("[embedding][error] batch item failed: %s", e)
                continue
        
        return embeddings
