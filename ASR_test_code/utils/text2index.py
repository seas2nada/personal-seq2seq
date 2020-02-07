import sys

# get token dictionary
def Dictionary(token_lines):
    dictionary = {}
    for line in token_lines:
        word = line.split('\t')[0]
        index = line.split('\t')[1].strip('\n')
        dictionary[word]=index

    return dictionary

# text sentences to token indexes
def main(text, tokens, index):

    # get data
    with open(text, 'r') as f:
        with open (tokens, 'r') as g:
            with open(index, 'w') as h:
                text_lines = f.readlines()
                token_lines = g.readlines()

                # get token dictionary
                dictionary = Dictionary(token_lines)

                # text to token indexes
                text_indexed = []
                for sentence in text_lines:
                    sentence = sentence.strip('\n').split()
                    sentence_idx = ""
                    for token in sentence:
                        sentence_idx += " " + dictionary[token]
                    sentence_idx = sentence_idx.strip() + '\n'

                    # save index
                    h.write(sentence_idx)


if __name__=="__main__":
    text = sys.argv[1]
    tokens = sys.argv[2]
    output = sys.argv[3]
    main(text, tokens, output)
    exit()