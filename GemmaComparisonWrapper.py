import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM, GenerationConfig

from image_utils import shuffle_image_dict


class GemmaComparisonWrapper:
    def __init__(self, device):
        self.MODEL_ID = "google/gemma-4-E2B-it"
        self.device = device
        self.processor = self.load_processor()
        self.model = self.load_model(self.device)

    def load_processor(self):
        processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        processor.image_processor.do_rescale = False
        return processor

    def load_model(self, device):
        model = AutoModelForMultimodalLM.from_pretrained(self.MODEL_ID, dtype="auto").to(device)
        for param in model.parameters():
            param.requires_grad = False
        return model

    def construct_comparison_prompt(self, shuffled_image_dict, comparison_question):
        messages = []
        messages.append({"role": "system", "content": "Your job is to decide which image you prefer."})
        comparison_prompt_content = []
        comparison_prompt_content.append(
            {"type": "text", "text": comparison_question + " Only say the number of the image."})
        for idx, img_pixels in enumerate(shuffled_image_dict):
            comparison_prompt_content.append({"type": "text", "text": f"Image {idx + 1}:"})
            comparison_prompt_content.append({"type": "image", "image": img_pixels['img_pixels']})
        messages.append({"role": "user", "content": comparison_prompt_content})
        return messages

    def prepare_inference(self, prompt, enable_thinking=True):
        inputs = self.processor.apply_chat_template(
            prompt,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        ).to(self.device)
        input_len = inputs["input_ids"].shape[-1]
        generation_config = GenerationConfig(
            max_new_tokens=4096,
            early_stopping=True,
            return_dict_in_generate=True,
            output_logits=True
        )
        return inputs, generation_config, input_len

    def decode_logit(self, logit):
        return self.processor.tokenizer.decode(logit)

    def get_preference_logits(self, next_token_logits, num_choices, shuffled_image_dict):
        token_ids = torch.tensor(
            [self.processor.tokenizer.vocab[f'{i + 1}'] for i in range(num_choices)],
            device=next_token_logits.device,
        )
        shuffled_logits = next_token_logits[token_ids]
        perm = torch.zeros(num_choices, dtype=torch.long, device=next_token_logits.device)
        for i, d in enumerate(shuffled_image_dict):
            perm[d['original_idx']] = i
        return shuffled_logits[perm]

    def compare_and_find_preferred_image(self, images, comparison_question):
        shuffled_image_dict = shuffle_image_dict(images)
        prompt = self.construct_comparison_prompt(shuffled_image_dict, comparison_question)
        inputs, generation_config, input_len = self.prepare_inference(prompt, enable_thinking=False)
        outputs = self.model(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            attention_mask=inputs["attention_mask"],
            image_position_ids=inputs.get("image_position_ids"),
            mm_token_type_ids=inputs.get("mm_token_type_ids"),
            return_dict=True,
        )

        next_token_logits = outputs.logits[0, -1, :]
        preference_logits = self.get_preference_logits(next_token_logits, len(images), shuffled_image_dict)
        predicted_token = self.processor.tokenizer.decode(torch.argmax(next_token_logits))
        if not predicted_token.isdigit():
            return -1, None, None
        preferred_image = int(predicted_token) - 1
        if preferred_image < 0 or preferred_image >= len(images):
            return -1, None, None
        return shuffled_image_dict[preferred_image]['original_idx'], preference_logits, next_token_logits

    def prompt_image_description(self, image):
        prompt = [
            {"role": "system", "content": "Your job is to describe the given image."},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "describe the given image"},
            ]
             }
        ]

        inputs = self.processor.apply_chat_template(
            prompt,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=True,
        ).to(self.device)
        input_len = inputs["input_ids"].shape[-1]
        generation_config = GenerationConfig(
            max_new_tokens=4096,
        )

        outputs = self.model.generate(**inputs, generation_config=generation_config)
        response = self.processor.decode(outputs[0][input_len:], skip_special_tokens=False)
        parsed_response = self.processor.parse_response(response)
        print(parsed_response['thinking'])
        print(parsed_response['content'])
