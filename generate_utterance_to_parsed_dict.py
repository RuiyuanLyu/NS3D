fewshot_examples = """
Input: the stool that is in the middle of the recycling bin and the backpack
Program: relate_multi(filter(scene(), stool), filter(scene(), recycling_bin), filter(scene(), backpack), middle)


Input: choose the cabinet that is close to the mirror
Program: relate(filter(scene(), cabinet), filter(scene(), mirror), close)


Input: select the cabinet that is farthest from the mirror
Program: relate(filter(scene(), cabinet), filter(scene(), mirror), farthest)


Input: looking at the front of the bed pick the telephone that is to the right of the bed
Program: relate_anchor(filter(scene(), telephone), filter(scene(), bed), filter(scene(), bed), right)


Input: the pillow that is far away from the sofa chair
Program: relate(filter(scene(), pillow), filter(scene(), sofa_chair), far)


Input: the person that is in the center of the desk and the nightstand
Program: relate_multi(filter(scene(), person), filter(scene(), desk), filter(scene(), nightstand), center)


Input: facing the front of the kitchen cabinets choose the door that is to the left of them
Program: relate_anchor(filter(scene(), door), filter(scene(), kitchen_cabinets), filter(scene(), kitchen_cabinets), left)


Input: choose the sink that is in the middle of the table and the mirror
Program: relate_multi(filter(scene(), sink), filter(scene(), table), filter(scene(), mirror), middle)


Input: choose the cabinet that is between the bookshelf and the desk
Program: relate_multi(filter(scene(), cabinet), filter(scene(), bookshelf), filter(scene(), desk), between)


Input: the stool that is between the bed and the kitchen cabinet
Program: relate_multi(filter(scene(), stool), filter(scene(), bed), filter(scene(), kitchen_cabinet), between)


Input: facing the front of the dresser pick the monitor that is on the left of the dresser
Program: relate_anchor(filter(scene(), monitor), filter(scene(), dresser), filter(scene(), dresser), left)


Input: select the table that is in front of the couch
Program: relate(filter(scene(), table), filter(scene(), couch), front)


Input: the shelf that is in the center of the chair and the radiator
Program: relate_multi(filter(scene(), shelf), filter(scene(), chair), filter(scene(), radiator), center)


Input: facing the front of the couch select the table that is on the left side of it
Program: relate_anchor(filter(scene(), table), filter(scene(), couch), filter(scene(), couch), left)


Input: choose the mirror that is in the middle of the paper towel dispenser and the shelf
Program: relate_multi(filter(scene(), mirror), filter(scene(), paper_towel_dispenser), filter(scene(), shelf), middle)


Input: looking the office chair from the side you sit on it choose the chair that is on the left side of this office chair
Program: relate_anchor(filter(scene(), chair), filter(scene(), office_chair), filter(scene(), office_chair), left)


Input: select the door that is farthest from the couch
Program: relate(filter(scene(), door), filter(scene(), couch), farthest)


Input: facing the front of the kitchen cabinets choose the cabinet that is on the right of them
Program: relate_anchor(filter(scene(), cabinet), filter(scene(), kitchen_cabinets), filter(scene(), kitchen_cabinets), right)


Input: find the mirror that is far away from the sink
Program: relate(filter(scene(), mirror), filter(scene(), sink), far)


Input: the stool that is far away from the telephone
Program: relate(filter(scene(), stool), filter(scene(), telephone), far)


Input: the trash can that is in the middle of the door and the cabinet
Program: relate_multi(filter(scene(), trash_can), filter(scene(), door), filter(scene(), cabinet), middle)


Input: select the shelf that is in front of the bookshelf
Program: relate(filter(scene(), shelf), filter(scene(), bookshelf), front)


Input: the chair that is in the center of the radiator and the bench
Program: relate_multi(filter(scene(), chair), filter(scene(), radiator), filter(scene(), bench), center)
"""

def make_message_from_fewshot_example(fewshot_examples):
    messages = [{"role": "system", "content": "You are a helpful assistant. You parse natural language instructions into codes."}]
    for example in fewshot_examples.split("\n\n"):
        example = example.strip()
        input_text, program_text = example.split("\n")
        input_text = input_text.replace("Input: ", "")
        program_text = program_text.replace("Program: ", "")
        messages.append({"role": "user", "content": input_text})
        messages.append({"role": "assistant", "content": program_text})
    return messages

FEWSHOT_MESSAGES = make_message_from_fewshot_example(fewshot_examples)

from openai_api import get_full_response
from copy import deepcopy

def parse_language(text):
    messages = deepcopy(FEWSHOT_MESSAGES)
    messages.append({"role": "user", "content": text})
    full_response = get_full_response(messages)
    value = full_response.choices[0].message.content.strip() 
    return {text: value}    

import os
import mmengine
def parse_language_batch(texts, save_path):
    if os.path.exists(save_path):
        return
    dict_list = mmengine.utils.track_parallel_progress(parse_language, texts, nproc=10)
    single_dict = {}
    for d in dict_list:
        single_dict.update(d)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(single_dict, f, ensure_ascii=False, indent=4)

source_file = "D:\Projects\shared_data\es_gen_data\VG.json"
temp_save_file_pattern = 'temp_files/temp_progress_{}.json'
target_file = "D:\Projects\shared_data\es_gen_data\VG_mapping.json"
target_pickle_file = "D:\Projects\shared_data\es_gen_data\VG_mapping.pkl"

import json
if __name__ == '__main__':
    with open(source_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    chunk_size = 100
    num_chunks = len(data) // chunk_size + 1
    for i in range(num_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        if end > len(data):
            end = len(data)
        chunk_data = data[start:end]
        chunk_texts = [d['text'] for d in chunk_data]
        temp_save_file = temp_save_file_pattern.format(i)
        print(f"Processing chunk {i+1}/{num_chunks} with {len(chunk_texts)} examples")
        parse_language_batch(chunk_texts, temp_save_file)
    # now combine the temp files
    mapping = {}
    for i in range(num_chunks):
        temp_save_file = temp_save_file_pattern.format(i)
        with open(temp_save_file, 'r', encoding='utf-8') as f:
            temp_mapping = json.load(f)
        mapping.update(temp_mapping)
    with open(target_file, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=4)
    import pickle
    with open(target_pickle_file, 'wb') as f:
        pickle.dump(mapping, f)
    print("Done")