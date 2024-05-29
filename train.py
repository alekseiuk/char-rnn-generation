import torch
import random
from data import *
from model import *


def random_training_example(category_lines, all_categories):
    
    def random_choice(a):
        random_idx = random.randint(0, len(a) - 1)
        return a[random_idx]
    
    category = random_choice(all_categories)
    line = random_choice(category_lines[category])
    category_tensor = category_to_tensor(category)
    input_tensor = input_to_tensor(line)
    target_tensor = target_to_tensor(line)
    return category_tensor, input_tensor, target_tensor


rnn = RNN(input_size=N_LETTERS, hidden_size=128, output_size=N_LETTERS, n_categories=n_categories)

criterion = nn.NLLLoss()
learning_rate = 0.0005

def train(category_tensor, input_tensor, target_tensor):
    target_tensor.unsqueeze_(-1)
    hidden = rnn.init_hidden()

    rnn.zero_grad()

    loss = torch.Tensor([0])

    for i in range(input_tensor.size(0)):
        output, hidden = rnn(category_tensor, input_tensor[i], hidden)
        l = criterion(output, target_tensor[i])
        loss += l

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item() / input_tensor.size(0)


if __name__ == '__main__':
    
    total_loss = 0
    all_losses = []
    print_steps = 5000
    n_iters = 100000

    for i in range(1, n_iters + 1):
        category_tensor, input_tensor, target_tensor = random_training_example(category_lines, all_categories)
        
        output, loss = train(category_tensor, input_tensor, target_tensor)
        total_loss += loss 
            
        if i % print_steps == 0:
            print(f"{i} {loss:.4f}")

    torch.save(rnn, 'char-rnn-generation.pt')