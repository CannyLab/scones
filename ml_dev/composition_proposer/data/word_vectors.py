import numpy as np

def get_glove(model, word, emb_size=300):
    if word in model:
        model[word] = list(model[word])
        return model[word]
    else:
        return [0.0] * emb_size

def get_glove_model(data_dir):
    glove_file = os.path.join(data_dir, "glove.42B.300d.txt")
    with open(glove_file, 'r') as f:
        model = {}
        for line in f:
            vals = line.rstrip().split(' ')
            model[vals[0]] = np.array(list(map(float, vals[1:])))
    return model    
