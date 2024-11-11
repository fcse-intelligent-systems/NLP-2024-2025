import numpy as np
from datasets import Dataset


def create_transformers_train_data(sentences, translations, tokenizer):
    inputs_en = tokenizer(sentences, max_length=10, truncation=True)

    with tokenizer.as_target_tokenizer():
        outputs_es = tokenizer(translations, max_length=10, truncation=True)

    data = Dataset.from_dict({'input_ids': inputs_en['input_ids'],
                              'attention_mask': inputs_en['attention_mask'],
                              'labels': outputs_es['input_ids']})
    return data


def decode_with_transformer(sentence, tokenizer, model):
    tokens = tokenizer([sentence], return_tensors='np')
    out = model.generate(**tokens, max_length=10)

    with tokenizer.as_target_tokenizer():
        pred_sentence = tokenizer.decode(out[0], skip_special_tokens=True)

    return pred_sentence
