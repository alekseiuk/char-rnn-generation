import torch
import string
import random
from data import *


rnn = torch.load('char-rnn-generation.pt')

max_length = 20

# Sample from a category and starting letter
def sample(category):
    start_letter = random.choice(string.ascii_uppercase)

    with torch.no_grad():  # no need to track history in sampling
        category_tensor = category_to_tensor(category)
        input = input_to_tensor(start_letter)
        hidden = rnn.init_hidden()

        output_name = start_letter

        for _ in range(max_length):
            output, hidden = rnn(category_tensor, input[0], hidden)
            idx = torch.argmax(output).item()
            if idx == N_LETTERS - 1:
                break
            else:
                letter = ALL_LETTERS[idx]
                output_name += letter
            input = input_to_tensor(letter)

        return output_name

# Get 3 samples from one category
def samples(category):
    print(f'Category: {category}')
    for _ in (range(3)):
        print(sample(category))


if __name__ == '__main__':
    samples('Russian')
    samples('German')
    samples('Spanish')
    samples('Chinese')