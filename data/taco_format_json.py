import os
import json
import argparse

def taco_format_json(input_json, output_json):
    """
    Convert a JSON file to a taco format JSON file.
    
    Args:
        input_json (str): Path to the input JSON file.
        output_json (str): Path to the output taco format JSON file.
    """
    template_json = '../storage_template.json'
    # load the JSON files
    with open(template_json, 'r') as f:
        taco_data = json.load(f)
    
    with open(input_json, 'r') as f:
        input_data = json.load(f)

    # Process scene_categories
    scene_categories = taco_data['scene_categories']
    
    # Add the scene_categories
    input_data.update({'scene_categories': scene_categories})
    # Update the category_ids
    template_ids = {category['name']: category['id'] for category in taco_data['categories']}
    categories = input_data['categories']
    # Update annotations
    for annotation in input_data['annotations']:
        name = categories[annotation['category_id']]["name"]
        annotation['category_id'] = template_ids[name]
    
    # Update category name definition
    input_data['categories'] = taco_data['categories']
    
    with open(output_json, 'w') as f:
        json.dump(input_data, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON to taco format")
    parser.add_argument("--input", required=True, type=str, help="Path to the input JSON file")
    parser.add_argument("--output", default="annotations_taco.json", type=str, help="Path to the output taco format JSON file")
    args = parser.parse_args()

    taco_format_json(args.input, args.output)