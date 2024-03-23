import numpy as np
import torch
from parameters import *

def generateCode(model, char2id, startSentence, limit=1000, temperature=1.):
    
    id2char = { val: char for char,val in char2id.items() }

    result = startSentence[1:]
    curr_single = startSentence[:1]

    generated = 0
    is_first_call = True
    model.eval()
    
    with torch.no_grad():
        
        while curr_single != '†' and generated < limit:
            X = model.preparePaddedBatch([list(curr_single)])
            E = model.embed(X)
            if is_first_call:
                output, (h, c) = model.lstm(E)
                is_first_call = False
            else:
                print(h.size())
                output, (h, c) = model.lstm(E, (h, c))
            
            z_wave = model.dropout(output)
            Z = model.projection(z_wave.flatten(0, 1))
            Z_prob = torch.nn.functional.softmax(Z / temperature, dim=1)
            Z_prob_last = Z_prob[-1,]
            index = Z_prob_last.multinomial(num_samples=1)
            curr_single = str(id2char[index[0].item()])
            result = result + curr_single
            generated += 1
            
    model.train()
    result = result[:-1]

	
    return result
