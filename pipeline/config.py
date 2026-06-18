import os
import yaml


class Config:
    def __init__(self, config_path: str = "config.yaml"):
        base_dir = os.path.dirname(__file__)
        full_config_path = os.path.join(base_dir, config_path)

        with open(full_config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        for key, value in config_data.items():
            setattr(self, key, value)

        self.dataset_path = os.path.join(base_dir, self.dataset_path)
        self.files_dir = os.path.join(base_dir, self.files_dir)
        self.cleaned_files_dir = os.path.join(base_dir, self.cleaned_files_dir)
        self.vector_store_path = os.path.join(base_dir, self.vector_store_path)

        if hasattr(self, "mlflow") and self.mlflow:
            if "artifact_location" not in self.mlflow:
                self.mlflow["artifact_location"] = os.path.join(base_dir, "mlruns_artifacts")
            else:
                self.mlflow["artifact_location"] = os.path.join(
                    base_dir, self.mlflow["artifact_location"]
                )
