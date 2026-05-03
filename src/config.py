from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    openai_api_key: str = "test-key"
    openai_model_dev: str = "gpt-4o-mini"
    openai_model_eval: str = "gpt-4.1"
    database_path: str = "./valura.db"
    max_response_timeout: int = 25
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
    
    def get_model_for_tier(self, tier: str) -> str:
        return self.openai_model_eval if tier == "premium" else self.openai_model_dev

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
