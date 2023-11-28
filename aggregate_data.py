filenames = {
            # 'gpt4v_1246k': "share-captioner_coco_lcs_sam_1246k_1107",
            #  'gpt4v_100k': "sharegpt4v_instruct_gpt4-vision_cap100k",
             'stage1_captions': "LLaVA-Pretrain/blip_laion_cc_sbu_558k"}
            #  'mix': "sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k"}

import os
from copy import deepcopy
def main():
    import json
    # all_name = "llava_v1_5_mix665k_flattened_multi_turn"
    all_names = ["llava_v1_5_mix665k", "llava_v1_5_mix665k_flattened_multi_turn"]
    # all_samples = json.load(open(f"./playground/data/{all_name}.json", 'r'))
    for all_name in all_names:
        for k, v in filenames.items():
            new_file = f"./playground/data/{all_name}_{k}.json"
            file = f"./playground/data/{v}.json"
            samples = json.load(open(file, 'r'))
            print(f"Total samples of {k}: {len(samples)}")

            mix_samples = json.load(open(f"./playground/data/{all_name}.json", 'r'))
            
            image_exists = 0
            for sample in samples:
                image_path = "./playground/data/" + sample["image"]
                if os.path.exists(image_path):
                    image_exists += 1
                    mix_samples.append(sample)
                    # all_samples.append(sample)
            
            print(f"Image exists: {image_exists}")
            print(f"Total samples of {k} after adding: {len(mix_samples)}")
            json.dump(mix_samples, open(new_file, 'w'), indent=4)
            print(f"Saved to {new_file}")
    # print(f"Total samples of all: {len(all_samples)}")
    # new_file = f"./playground/data/{all_name}_all.json"
    # json.dump(all_samples, open(new_file, 'w'), indent=4)
    
if __name__ == "__main__":
    main()