import os
import spacy
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
    