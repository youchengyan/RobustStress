import spacy
import json
import random
import os
from tqdm import tqdm
import argparse
from nltk.corpus import wordnet
import nltk

try:
    wordnet.synsets('test')
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)


def get_synonym(word, pos):
    """Based on the part-of-speech tags from spaCy, return a synonym (if one exists); otherwise, return the original word."""
    pos_map = {
        'ADJ': wordnet.ADJ,
        'ADV': wordnet.ADV,
    }
    wn_pos = pos_map.get(pos)
    if not wn_pos:
        return word

    synsets = wordnet.synsets(word, pos=wn_pos)
    if not synsets:
        return word

    lemmas = set()
    for syn in synsets:
        for lemma in syn.lemmas():
            name = lemma.name().replace('_', ' ')
            if name.lower() != word.lower():
                lemmas.add(name)

    if lemmas:
        return random.choice(list(lemmas))
    else:
        return word


def main():
    parser = argparse.ArgumentParser(description="Replace p% of ADJ/ADV tokens with synonyms using WordNet.")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing 'test.json'")
    args = parser.parse_args()

    random.seed(42)
    nlp = spacy.load('en_core_web_sm')

    input_path = os.path.join(args.input_dir, "test.json")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [line["text"] for line in data]
    labels = [line["label"] for line in data]

    output_dir = os.path.join(args.input_dir, "SynonymReplacement")
    os.makedirs(output_dir, exist_ok=True)

    L = [5, 10, 15, 20]

    for p in L:
        random.seed(42)
        result = []

        for i in tqdm(range(len(texts)), desc=f"Processing p={p}%"):
            text = texts[i]
            doc = nlp(text)

            tokens = []
            candidate_indices = []

            for idx, token in enumerate(doc):
                tokens.append(token.text)
                if token.pos_ in ("ADJ", "ADV"):
                    candidate_indices.append(idx)

            if not candidate_indices:
                new_text = text
            else:
                # the num of replacement
                num_to_replace = round(len(candidate_indices) * p / 100)
                num_to_replace = min(num_to_replace, len(candidate_indices))

                if num_to_replace == 0:
                    new_text = text
                else:
                    selected_indices = set(random.sample(candidate_indices, num_to_replace))
                    for idx in selected_indices:
                        orig_word = tokens[idx]
                        pos = doc[idx].pos_
                        synonym = get_synonym(orig_word, pos)
                        tokens[idx] = synonym
                    new_text = " ".join(tokens)

            result.append({"text": new_text, "label": labels[i]})

        output_path = os.path.join(output_dir, f"test_perturb_{p}.json")
        with open(output_path, "w", encoding="utf-8") as w:
            json.dump(result, w, ensure_ascii=False, indent=2)

    print("✅ Synonym replacement (ADJ + ADV only) completed.")


if __name__ == "__main__":
    main()