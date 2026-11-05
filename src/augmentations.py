import random
import numpy as np

def rotate_grid(grid, k):
    return np.rot90(np.array(grid), k).tolist()

def flip_grid(grid, mode='horizontal'):
    arr = np.array(grid)
    if mode == 'horizontal':
        return np.fliplr(arr).tolist()
    return np.flipud(arr).tolist()

def permute_colors(grid, color_map):
    return [[color_map.get(cell, cell) for cell in row] for row in grid]

def generate_augmentations(train_pairs, num_augmentations=30):
    """Generates augmented input-output training pairs for Test-Time Training."""
    augmented = [(p['input'], p['output']) for p in train_pairs]
    while len(augmented) < num_augmentations:
        pair = random.choice(train_pairs)
        inp, out = pair['input'], pair['output']
        op = random.choice(['rot', 'flip', 'color'])
        
        if op == 'rot':
            k = random.choice([1, 2, 3])
            aug_in, aug_out = rotate_grid(inp, k), rotate_grid(out, k)
        elif op == 'flip':
            mode = random.choice(['horizontal', 'vertical'])
            aug_in, aug_out = flip_grid(inp, mode), flip_grid(out, mode)
        else:
            colors = list(range(10))
            shuffled = colors.copy()
            random.shuffle(shuffled)
            cmap = dict(zip(colors, shuffled))
            aug_in, aug_out = permute_colors(inp, cmap), permute_colors(out, cmap)
            
        augmented.append((aug_in, aug_out))
    return augmented[:num_augmentations]