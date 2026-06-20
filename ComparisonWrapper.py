from abc import ABC, abstractmethod

from transformers import AutoProcessor, AutoModelForMultimodalLM


class ComparisonWrapper(ABC):
    @abstractmethod
    def load_processor_and_model(self) -> (AutoProcessor, AutoModelForMultimodalLM):
        pass

    @abstractmethod
    def compare_and_find_preferred_image(self, image_paths, processor, model) -> int:
        pass