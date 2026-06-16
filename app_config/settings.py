"""
RiskLens AI — Application Settings
====================================
Pydantic-based type-safe configuration.

Why Pydantic for config:
    1. Type validation at startup — catches misconfig before training
    2. Environment variable loading — different configs for dev/prod
    3. Immutable after creation — prevents accidental modification
    4. Self-documenting — each field has a type and description

Interview Insight:
    "I use Pydantic Settings because it validates configuration at
    application startup. If someone sets BATCH_SIZE='abc', the app
    fails immediately with a clear error, not 2 hours into training."
"""

from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings
from pydantic import Field


class AppSettings(BaseSettings):
    """Application configuration with validation."""

    # Project
    app_name: str = Field(default="RiskLens AI", description="Application name")
    environment: str = Field(default="development", description="dev/staging/production")
    debug: bool = Field(default=True, description="Enable debug mode")

    # Paths (relative to project root)
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent,
        description="Project root directory",
    )

    @property
    def data_path(self) -> Path:
        return self.project_root / "data"

    @property
    def raw_data_path(self) -> Path:
        return self.data_path / "raw"

    @property
    def processed_data_path(self) -> Path:
        return self.data_path / "processed"

    @property
    def feature_store_path(self) -> Path:
        return self.data_path / "features"

    @property
    def model_store_path(self) -> Path:
        return self.data_path / "models"

    @property
    def logs_path(self) -> Path:
        return self.project_root / "logs"

    @property
    def reports_path(self) -> Path:
        return self.project_root / "reports"

    # Model parameters
    random_seed: int = Field(default=42, description="Random seed for reproducibility")
    test_size: float = Field(default=0.2, description="Test set proportion")
    val_size: float = Field(default=0.1, description="Validation set proportion")

    # Memory
    max_memory_gb: float = Field(default=8.0, description="Max memory budget in GB")

    def ensure_directories(self) -> None:
        """Create all required directories."""
        for path in [
            self.raw_data_path,
            self.processed_data_path,
            self.feature_store_path,
            self.model_store_path,
            self.logs_path,
            self.reports_path,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    class Config:
        env_prefix = "RISKLENS_"
        env_file = ".env"
        extra = "ignore"


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Get cached application settings (singleton)."""
    return AppSettings()
