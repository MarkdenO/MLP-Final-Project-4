# Machine Learning Project Group 4
# Authors: Julian Paagman, Pascal Boon, Niek Biesterbos & Mark den Ouden


import os


def preprocess_data():
    """
       Lowercase everything, except named entities.
       Change emojis to <emoji> token.
       Change punctuation to <punct> token.
       Change numbers to <num> token. 
    """
    data_path = 'lid_spaeng'
    files = os.listdir(data_path)
    for file in files[:2]:
        # File extension must be .conll
        if not file.endswith('.conll'):
            continue
        with open(data_path + '/' + file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                print(line, end='')

def main():
    # Preprocess the data
    data = preprocess_data()

if __name__ == '__main__':
    main()