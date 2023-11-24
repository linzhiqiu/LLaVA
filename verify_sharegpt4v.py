filenames = {'1246k': "share-captioner_coco_lcs_sam_1246k_1107",
             '100k': "sharegpt4v_instruct_gpt4-vision_cap100k",
             'mix': "sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k"}

import os
from copy import deepcopy
def main():
    import json
    for k, v in filenames.items():
        file = f"./playground/data/{v}.json"
        samples = json.load(open(file, 'r'))
        print(f"Total samples of {k}: {len(samples)}")

        image_exists = 0
        for sample in samples:
            image_path = "./playground/data/" + sample["image"]
            if os.path.exists(image_path):
                image_exists += 1
                
        print(f"Image exists: {image_exists}")
    
if __name__ == "__main__":
    main()