import os
import random
import json
import argparse
import numpy as np
from time import time
from os.path import join, exists
from nltk import word_tokenize, sent_tokenize
from datetime import timedelta

# tokenize a turn
def tokenize(sent):
    return ' '.join(word_tokenize(sent))

# return the number of the tokens in a long dialogue
def token_num(data):
    cnt = 0
    for i in range(len(data)):
        cnt += len(data[i].split())
    return cnt

# find the last possible position of the starting point of the noised window
def last_position(data, window_size):
   n = len(data)
   cnt = 0
   for i in range(n - 1, -1, -1):
        cnt += len(data[i].split())
        if cnt > window_size:
            return i

def find_str(s, s_target):
    for i in range(len(s)):
        if s[i] == s_target:
            return i

def turn_split(data):
    new_data = []
    lens = []
    for i in range(len(data)):
        lens.append(len(data[i].split()))
    idx = lens.index(max(lens))
    for i in range(len(data)):
        if i == idx:
            split_turn = sent_tokenize(data[i])
            for j in range(len(split_turn)):
                if j == 0:
                    new_data.append(split_turn[j])
                else:
                    cur_s = '[MASK] : ' + split_turn[j]
                    new_data.append(cur_s)
        else:
            new_data.append(data[i])
    return new_data

def turn_merge(data):
    new_data = []
    n_turns = len(data)
    st_idx = random.randint(0, n_turns - 2) # merge at least 2 turns
    merge_turns = max(2, np.random.poisson(3, 1)[0])
    ed_idx = min(n_turns - 1, st_idx + merge_turns - 1)
    new_turn = []
    for i in range(len(data)):
        if i >= st_idx and i <= ed_idx:
            if i == st_idx:
                new_turn.append(data[i])
            else:
                s = data[i].split()
                idx = find_str(s, ':')
                cur = []
                for j in range(idx + 1, len(s)):
                    cur.append(s[j])
                new_turn.append(' '.join(cur))
            if i == ed_idx:
                new_data.append(' '.join(new_turn))
        else:
            new_data.append(data[i])
    return new_data

def speaker_mask(data, speaker_mask_ratio):
    new_data = []
    for i in range(len(data)):
        s = data[i].split()
        cur = []
        if random.random() < speaker_mask_ratio:
            idx = find_str(s, ':')
            cur.append('[MASK]')
            for j in range(idx, len(s)):
                cur.append(s[j])
        else:
            for j in range(len(s)):
                cur.append(s[j])
        new_data.append(' '.join(cur))
    return new_data

def text_infilling(data, text_infilling_ratio):
    new_data = []
    for i in range(len(data)):
        s = data[i].split()
        cur = []
        left = 0
        for j in range(len(s)):
            if left > 0:
                left = left - 1
                continue
            if s[j] == '[MASK]':
                cur.append(s[j])
                continue
            if random.random() < text_infilling_ratio:
                left = np.random.poisson(3, 1)[0]
                cur.append('[MASK]')
                if left > 0:
                    left = left - 1
                    continue
            cur.append(s[j])
        new_data.append(' '.join(cur))
    return new_data

def shuffling(data):
    idx = list(range(len(data)))
    random.shuffle(idx)
    new_data = []
    for i in range(len(idx)):
        new_data.append(data[idx[i]])
    return new_data

# [start]: the start point of the noised window
# [end]: the end point of the noised window
def add_special_token(data):
    data[0] = '[start] ' + data[0]
    data[len(data) - 1] = data[len(data) - 1] + ' [end]'
    return data

def add_noise(data, window_size):
    st_idx = 0
    ed_idx = last_position(data, window_size)
    # the start point of the window
    window_st = random.randint(st_idx, ed_idx)
    # the end point of the window
    window_ed = window_st
    # the number of words in the window
    window_cnt = 0
    window_data = []
    for i in range(window_st, len(data)):
        if window_cnt + len(data[i].split()) > window_size:
            break
        window_cnt += len(data[i].split())
        window_data.append(data[i])
        window_ed = i
    # can not find a suitable window in this dialogue
    if len(window_data) == 0:
        return -1, -1
    # first speaker mask
    noise_data = speaker_mask(window_data, args.speaker_mask_ratio)
    # then turn_split or turn_merge
    if len(noise_data) == 1:
        # If there is only one turn in the window, only turn split can be performed
        noise_data = turn_split(noise_data)
    else:
        # Randomly choose turn split or turn merge
        if random.random() < args.turn_split_prob:
            noise_data = turn_split(noise_data)
        else:
            noise_data = turn_merge(noise_data)
    # then text infilling and turn shuffling
    noise_data = text_infilling(noise_data, args.text_infilling_ratio)
    noise_data = shuffling(noise_data)
    # add [start] and [end]
    noise_data = add_special_token(noise_data)
    source = []
    have_window = 0
    for i in range(len(data)):
        if i >= window_st and i <= window_ed:
            if have_window == 0:
                for j in range(len(noise_data)):
                    source.append(noise_data[j])
                have_window = 1
        else:
            source.append(data[i])
    return source, window_data

def to_json(src, tgt):
    cur = {}
    cur['src'] = ' '.join(src)
    cur['tgt'] = ' '.join(tgt)
    return cur

def write_jsonl(data, path):
    processed_path = join(join(path, '..'), 'processed_dialogues')
    if not exists(processed_path):
        os.makedirs(processed_path)
    dialogue_path = join(processed_path, 'dialogue_with_noised_window.jsonl')
    assert not exists(dialogue_path)
    with open(dialogue_path, 'w') as f:
        for i in range(len(data)):
            print(json.dumps(data[i], ensure_ascii=False), file=f)

def main(args):
    
    dialogues = os.listdir(args.data_path)
    n = len(dialogues)
    print('Total of {} dialogues to be processed'.format(n))

    start = time()
    all_dialogues = []
    for i in range(n):
        dialogue = dialogues[i]
        data = []
        with open(join(args.data_path, dialogue)) as f:
            for line in f:
                data.append(tokenize(line.strip()))
        cnt = token_num(data)
        src, tgt = add_noise(data, min(args.max_window_words, 
                                       int(cnt * args.window_ratio)))
        if src == -1 and tgt == -1:
            continue
        all_dialogues.append(to_json(src, tgt))
        print('{}/{} ({:.2f}%) processed in {} seconds\r'.format(
               i, n, i/n*100, timedelta(seconds=int(time()-start))), end='')

    print('\nFinished data processing and start writing data !!!')
    write_jsonl(all_dialogues, args.data_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Randomly select a windows in a long dialogue \
                     and inject five different types of noise'
    )

    parser.add_argument('--data_path', default='data/dialogues',
        help='Path to the long dialogue data', type=str)
    parser.add_argument('--window_ratio', default=0.2,
        help='Ratio of the original dialogue as a noised window', type=float)
    parser.add_argument('--max_window_words', default=450,
        help='Maximum number of words in the window', type=int)
    parser.add_argument('--speaker_mask_ratio', default=0.5,
        help='Ratio of speakers in the window that will be masked', type=float)
    parser.add_argument('--text_infilling_ratio', default=0.1,
        help='Ratio of text infilling in the window', type=float)
    parser.add_argument('--turn_split_prob', default=0.5,
        help='Probability of choosing turn split, 1 - turn_split_prob \
              is the probability of turn merge', type=float)

    args = parser.parse_args()

    main(args)