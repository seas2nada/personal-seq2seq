# get token dictionary
def Dictionary(dict_dir):
    dictionary = []
    with open(dict_dir, 'r') as f:
        dicts = f.readlines()
        for dict in dicts:
            dictionary.append(str(dict.split('\t')[0]))

    return dictionary

# convert index token sentence to string sentence
def index_to_text(sentence):
    dictionary = Dictionary("data/lang/tokens.txt")

    sentence = list(map(int, sentence))

    for i, token in enumerate(sentence):
        sentence[i] = dictionary[token]

    sentence = "".join(sentence)
    sentence = sentence.replace("▁", " ")
    return sentence