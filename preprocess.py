import os
import spacy
import re
nlp = spacy.load("en_core_web_sm")

from collections import defaultdict


def parse_sentences(text):
    """
        Parse the CoNLL-U format into a dictionary of sentences.
        Format of dictionary: {sentence_number: [(word, label), ...], ...}
    """
    sentences = defaultdict(list)
    current_sent = None

    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        if line.startswith("# sent_enum ="):
            current_sent = int(line.split("=")[1].strip())
        else:
            word, label = line.rsplit(maxsplit=1)
            sentences[current_sent].append((word, label))

    return dict(sentences)


def get_named_entities(parsed_sentences):
    """
        Identify Named Entities in the data.
    """
    # Create sentences from the dictionary
    # Process each sentence with spaCy for Named Entity Recognition (NER)
    entities_per_sentence = {}

    for i in parsed_sentences.items():
        for file, sentence in parsed_sentences.items():
            for sent_id, sent in sentence.items():
                # Put every sentence together and identify NE
                sentence = " ".join([word for word, label in sent])
                doc = nlp(sentence)
                entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ["PER", "LOC", "ORG"]]
                entities_per_sentence[sent_id] = entities
            
    print(entities_per_sentence.keys())
    # Print results
    for sent_id in entities_per_sentence.keys():
        # print(sent_id)
        # print(parsed_sentences)
        print(f"Sentence {sent_id}: {entities_per_sentence[sent_id]}")
        if entities_per_sentence[sent_id]:
            print(f"{sent_id}: Named Entities:")
            for entity, label in entities_per_sentence[sent_id]:
                print(f"    {entity} -> {label}")
        else:
            print(f"{sent_id}:  No named entities found.")
        print()


def preprocess_emojis(sentence_dict):
    """
        Identify emojis in the data.
        Change emojis to <emoji> token.
    """
    emojis_re = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons 😀-🙏
        "\U0001F300-\U0001F5FF"  # Misc symbols & pictographs 🌀-🗿
        "\U0001F680-\U0001F6FF"  # Transport & map 🚀-🛶
        "\U0001F700-\U0001F77F"  # Alchemical symbols ⚗️-🝿
        "\U0001F780-\U0001F7FF"  # Geometric shapes 🞀-🟿
        "\U0001F800-\U0001F8FF"  # Supplemental arrows 🠀-🢿
        "\U0001F900-\U0001F9FF"  # Supplemental symbols & pictographs 🤀-🧿
        "\U0001FA00-\U0001FA6F"  # Chess symbols & more 🨀-🩯
        "\U0001FA70-\U0001FAFF"  # Extended pictographs 🩰-🫿
        "\U00002702-\U000027B0"  # Dingbats ✂-➰
        "\U000024C2-\U0001F251"  # Enclosed characters Ⓜ-🉑
        "]+",
        flags=re.UNICODE
    )

    # Iterate over all sentences and change emojis to <emoji> token
    for file, sentences in sentence_dict.items():
        for sent_id, sent in sentences.items():
            sentence = [(emojis_re.sub("<emoji>", word), label) for word, label in sent]
            sentence_dict[file][sent_id] = sentence
    
    return sentence_dict


def preprocess_data(data_path):
    """
       Lowercase everything, except named entities.
       Change emojis to <emoji> token.
       Change punctuation to <punct> token.
       Change numbers to <num> token. 
    """
    files = os.listdir(data_path)
    sentence_dict = defaultdict()
    for file in files[:1]:
        # File extension must be .conll
        if not file.endswith('.conll'):
            continue
        with open(data_path + '/' + file, 'r', encoding='utf-8') as f:
            # Parse the CoNLL-U format into a dictionary of sentences.
            sentence_dict[file] = parse_sentences(f.read())

    # Identify Named Entities
    # Doesn't really work
    # named_entities = get_named_entities(sentence_dict)

    # Lowercase everything, except named entities
    named_entities = []
    
    # Lowercase and store in variable
    for file, sentences in sentence_dict.items():
        for sent_id, sent in sentences.items():
            sentence = [(word.lower(), label) if label not in named_entities else (word, label) for word, label in sent]
            sentence_dict[file][sent_id] = sentence
    # Identify emojis
    # Change emojis to <emoji> token
    sentence_dict = preprocess_emojis(sentence_dict)
    print(sentence_dict.items())
