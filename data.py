import os
import torch
import glob
import unicodedata
import string

# alphabet small + capital letters + " .,;'-"
ALL_LETTERS = string.ascii_letters + " .,;'-"
N_LETTERS = len(ALL_LETTERS) + 1 # Plus EOS marker


# Turn a Unicode string to plain ASCII
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in ALL_LETTERS
    )

# Read a file and split into lines
def load_data():
    category_lines = {}
    all_categories = []
    
    def find_files(path):
        return glob.glob(path)
    
    # Read a file and split into lines
    def read_lines(filename):
        with open(filename, encoding='utf-8') as some_file:
            return [unicode_to_ascii(line.strip()) for line in some_file]
    
    # Build the category_lines dictionary, a list of names per language
    for filename in find_files('data/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        
        lines = read_lines(filename)
        category_lines[category] = lines
    
    n_categories = len(all_categories)

    if n_categories == 0:
        raise RuntimeError('Data not found. Make sure that you downloaded data '
                           'from https://download.pytorch.org/tutorial/data.zip and extract it to '
                           'the current directory.')

    return category_lines, all_categories, n_categories


category_lines, all_categories, n_categories = load_data()


def category_to_tensor(category):
    i = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][i] = 1
    return tensor

def letter_to_index(letter):
    return ALL_LETTERS.find(letter)

def input_to_tensor(line):
    tensor = torch.zeros(len(line), 1, N_LETTERS)
    for i, letter in enumerate(line):
        tensor[i][0][letter_to_index(letter)] = 1
    return tensor

def target_to_tensor(line):
    letter_indexes = [letter_to_index(line[i]) for i in range(1, len(line))]
    letter_indexes.append(N_LETTERS - 1) # EOS
    return torch.LongTensor(letter_indexes)