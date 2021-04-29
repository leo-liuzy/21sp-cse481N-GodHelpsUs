from pathlib import Path
import json


def read_amazon_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    langs = []
    labels = []
    for text_file in split_dir.iterdir():
        file_path = str(text_file)
        assert "json" in file_path
        with open(text_file, "r") as in_file:
            lines = in_file.readlines()
            for line in lines:
                if line.strip() == "":
                    continue
                line_obj = json.loads(line)
                assert all(x in line_obj for x in ["language", "review_body", "stars"])
                langs.append(line_obj["language"])
                texts.append(line_obj["review_body"])
                labels.append(line_obj["stars"])
    return texts, langs, labels


train_texts, train_langs, train_labels = read_amazon_split('amazon-reviews-ml/train')
test_texts, test_langs, test_labels = read_amazon_split('amazon-reviews-ml/test')

