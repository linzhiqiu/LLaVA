import os

def main():
    import json
    file = "./playground/data/llava_v1_5_mix665k.json"
    samples = json.load(open(file, 'r'))
    print(f"Total samples: {len(samples)}")
    
    single_turn_samples = []
    truncated_single_turn_samples = []
    for sample in samples:
        if len(sample["conversations"]) == 2:
            single_turn_samples.append(sample)
        sample["conversations"] = sample["conversations"][:2]
        truncated_single_turn_samples.append(sample)
        
    print(f"Total single-turn samples: {len(single_turn_samples)}")
    new_file = "./playground/data/llava_v1_5_mix665k_single_turn.json"
    json.dump(single_turn_samples, open(new_file, 'w'), indent=4)
    print(f"Saved to {new_file}")
    
    new_truncated_file = "./playground/data/llava_v1_5_mix665k_single_turn_truncated.json"
    new_truncated_file = "./playground/data/llava_v1_5_mix665k_single_turn_truncated.json"
    json.dump(truncated_single_turn_samples, open(new_truncated_file, 'w'), indent=4)
    print(f"Saved to {new_truncated_file}")
    
if __name__ == "__main__":
    main()