# Dialogue-Inspired Noise
We release the data preprocessing code here, i.e. how to inject five different types of noise into a long conversation.

## Data Format
The data format of original long dialogues should follow the text files in `data/dialogues`. Specifically, each text file (e.g., `data/dialogues/0.txt`) represents a long dialogue, where each line represents a turn. the format should look like:
```
Speaker 1: xxx ...
Speaker 2: yyy ...
Speaker 3 (or 1): zzz ...
...
```

## Data Preprocessing

```bash
python inject_noise_to_dialogue.py
```
The processed dialogues will be stored in `data/processed_dialogues/dialogue_with_noised_window.jsonl`.
- *src* indicates the conversation with the noised window.
- *tgt* is the original content of the window.
