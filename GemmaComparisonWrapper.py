import random

from transformers import AutoProcessor, AutoModelForMultimodalLM, GenerationConfig

from ComparisonWrapper import ComparisonWrapper


class GemmaComparisonWrapper(ComparisonWrapper):
    def __init__(self, device):
        self.MODEL_ID = "google/gemma-4-E2B-it"
        self.device = device

    def load_processor(self):
        return AutoProcessor.from_pretrained(self.MODEL_ID)

    def load_model(self, device):
        return AutoModelForMultimodalLM.from_pretrained(self.MODEL_ID, dtype="auto").to(device)

    def load_processor_and_model(self):
        processor = self.load_processor()
        model = self.load_model(self.device)
        return processor, model

    def shuffle_image_dict(self, image_paths):
        shuffled_image_dict = []
        for idx, img in enumerate(image_paths):
            shuffled_image_dict.append({'original_idx' : idx, 'img_path' : img})

        random.shuffle(shuffled_image_dict)
        return shuffled_image_dict

    def construct_comparison_prompt(self, shuffled_image_dict):
        messages = []
        messages.append({"role": "system", "content": "Your job is to decide which image you prefer."})
        comparison_prompt_content = []
        for idx, img_path in enumerate(shuffled_image_dict):
            comparison_prompt_content.append({"type": "text", "text": "Image " + str(idx + 1) + ":"})
            comparison_prompt_content.append({"type": "image", "path": img_path['img_path']})
        comparison_prompt_content.append({"type": "text", "text": "Which image do you prefer most? Only say the number of the image."})
        messages.append({"role": "user", "content": comparison_prompt_content})
        return messages

    def prepare_inference(self, prompt, processor):
        inputs = processor.apply_chat_template(
            prompt,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=True,
        ).to(self.device)
        input_len = inputs["input_ids"].shape[-1]
        print(input_len)
        generation_config = GenerationConfig(max_new_tokens=1024, early_stopping=True)
        return inputs, generation_config, input_len

    def compare_and_find_preferred_image(self, image_paths, processor, model):
        shuffled_image_dict = self.shuffle_image_dict(image_paths)
        prompt = self.construct_comparison_prompt(shuffled_image_dict)
        inputs, generation_config, input_len = self.prepare_inference(prompt, processor)
        outputs = model.generate(**inputs, generation_config=generation_config)
        response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
        content = processor.parse_response(response)["content"]
        preferred_image = int(content) - 1
        return shuffled_image_dict[preferred_image]['original_idx']