import json
import argparse
import colorsys
from xml.etree import ElementTree as ET
from xml.dom import minidom
from collections import defaultdict

def generate_supercategory_colors(supercategories):
    """Generate colors for supercategories and their subcategories."""
    color_map = {}
    num_super = len(supercategories)
    
    # Generate evenly-distributed colors for each supercategory
    for idx, (supercategory, subcategories) in enumerate(supercategories.items()):
        base_hue = idx / num_super
        
        # Generate colors for subcategories
        sub_colors = []
        num_sub = len(subcategories)
        for sub_idx in range(num_sub):
            lightness = 0.4 + 0.2 * (sub_idx % 3)  # 0.4-0.8
            saturation = 0.6 + 0.2 * (sub_idx % 2)  # 0.6-0.8
            rgb = colorsys.hls_to_rgb(base_hue, lightness, saturation)
            hex_color = "#{:02x}{:02x}{:02x}".format(
                int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
            sub_colors.append(hex_color)
        
        color_map[supercategory] = sub_colors
    
    return color_map

def json_to_labelstudio_xml(input_json, output_xml):
    # Read JSON file
    with open(input_json, 'r') as f:
        data = json.load(f)
    print(data)
    view = ET.Element('View')
    ET.SubElement(view, 'Header', value="Image Annotation For Trash Classification")
    ET.SubElement(view, "Header",value="Please classify the image, draw the regions, and draw bounding boxes around the trash items.")
    ET.SubElement(view, 'Image', name="image", value="$image", zoom="true")
    
    supercategories = defaultdict(list)
    for category in data['categories']:
        supercategories[category['supercategory']].append(category)
    
    color_map = generate_supercategory_colors(supercategories)
    
    poly_labels = ET.SubElement(view, 'PolygonLabels', 
                                  name=f"polygon", 
                                  toName="image",
                                  showInline="true",
                                  required="false",
                                  maxUsages="0",
                                  strokeWidth="3", 
                                  pointSize="small",
                                  opacity="0.9")
    
    rect_labels = ET.SubElement(view, 'RectangleLabels',
                                  name=f"bbox", 
                                  toName="image",
                                  showInline="true",
                                  required="false",
                                  maxUsages="0",
                                  strokeWidth="3", 
                                  pointSize="small",
                                  opacity="0.9")
    for supercategory, subcategories in supercategories.items():
        poly_view = ET.SubElement(poly_labels, 'View', style="marginRight: 20px")
        ET.SubElement(poly_view, 'Header', value=f"{supercategory}")
        rect_view = ET.SubElement(rect_labels, 'View', style="marginRight: 20px")
        ET.SubElement(rect_view, 'Header', value=f"{supercategory}")
        colors = color_map[supercategory]
        for idx, category in enumerate(subcategories):
            ET.SubElement(poly_view, 'Label',
                         value=category['name'],
                         background=colors[idx])
            ET.SubElement(rect_view, 'Label',
                         value=category['name'],
                         background=colors[idx])
    
    scene_header = ET.SubElement(view, 'Header', value="Scene Classification")
    choices = ET.SubElement(view, 'Choices', name="scene", toName="image", choice="single")
    for scene in data['scene_categories']:
        ET.SubElement(choices, 'Choice', value=scene['name'])
    # ET.SubElement(view, 'AutoSave', interval="60")
    # keymap = ET.SubElement(view, 'KeyMap', name="KeyMap", toName="image")
    # ET.SubElement(keymap, 'Key', value="a", action="accept")
    xml_str = minidom.parseString(ET.tostring(view)).toprettyxml(indent="  ")
    
    with open(output_xml, 'w', encoding='utf-8') as f:
        f.write(xml_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Label Studio XML template from JSON.')
    parser.add_argument('--input', required=True, help='Input JSON file path')
    parser.add_argument('--output', default='labelstudio_template.xml', help='Output XML file path')
    args = parser.parse_args()

    try:
        json_to_labelstudio_xml(args.input, args.output)
        print(f"Template generated: {args.output}")
    except Exception as e:
        print(f"Error: {str(e)}")