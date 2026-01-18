import xml.etree.ElementTree as ET
from collections import OrderedDict
from xml.dom import minidom


def get_sionna_material_id(original_id):
    """
    Maps original material IDs to their Sionna ITU material IDs.
    Returns both the new ID and the ITU material name.
    """
    mapping = {
        'mat-wall': ('mat-itu_concrete', 'itu-concrete'),
        'mat-roof': ('mat-itu_concrete', 'itu-concrete'),
        'mat-vegetation': ('mat-itu_wet_ground', 'itu-wet_ground'),
        'mat-forest': ('mat-itu_wet_ground', 'itu-wet_ground'),
        'mat-roads_residential': ('mat-itu_concrete', 'itu-concrete'),
        'mat-paths_footway': ('mat-itu_very_dry_ground', 'itu-very_dry_ground'),
        'mat-roads_track': ('mat-itu_medium_dry_ground', 'itu-medium_dry_ground'),
        'mat-roads_service': ('mat-itu_concrete', 'itu-concrete'),
    }
    
    if original_id in mapping:
        return mapping[original_id]
    else:
        # Default fallback for any other mat-* material
        return (f"mat-itu_concrete", 'itu-concrete')


def create_sionna_bsdf_element(new_id, itu_material_name):
    """
    Creates a simple Sionna-compatible BSDF element for the given ITU material.
    """
    # Create a simple diffuse BSDF (Sionna will replace this with the actual radio material)
    bsdf = ET.Element('bsdf', {
        'type': 'twosided',
        'id': new_id,
        'name': new_id
    })
    
    # Inner diffuse BSDF
    inner = ET.SubElement(bsdf, 'bsdf', {
        'type': 'diffuse',
        'name': 'bsdf'
    })
    
    # Default gray color for visualization
    ET.SubElement(inner, 'rgb', {
        'name': 'reflectance',
        'value': '0.800000 0.800000 0.800000'
    })
    
    return bsdf


def process_xml_file(input_file, output_file):
    """
    Processes the XML file:
    1. Replaces all material IDs with Sionna equivalents
    2. Removes duplicate BSDF declarations
    3. Updates all references in the file
    """
    print(f"Processing XML file: {input_file}")
    
    # Parse the XML file
    tree = ET.parse(input_file)
    root = tree.getroot()
    
    # Track mappings from old IDs to new IDs
    id_mappings = {}
    
    # Track unique Sionna BSDFs by their new ID
    unique_sionna_bsdfs = OrderedDict()
    bsdfs_to_remove = []
    
    # First pass: process all BSDF elements
    all_bsdfs = root.findall('.//bsdf')
    
    for bsdf in all_bsdfs:
        original_id = bsdf.get('id')
        
        # Only process materials starting with 'mat-'
        if not original_id or not original_id.startswith('mat-'):
            continue
        
        # Get the Sionna equivalent
        new_id, itu_material_name = get_sionna_material_id(original_id)
        id_mappings[original_id] = (new_id, itu_material_name)
        
        # Check if we've already created this Sionna BSDF
        if new_id not in unique_sionna_bsdfs:
            # Create the Sionna BSDF and store its position
            sionna_bsdf = create_sionna_bsdf_element(new_id, itu_material_name)
            parent = root
            index = list(parent).index(bsdf)
            
            unique_sionna_bsdfs[new_id] = {
                'element': sionna_bsdf,
                'original_element': bsdf,
                'parent': parent,
                'index': index,
                'itu_material_name': itu_material_name
            }
        else:
            # Mark this duplicate for removal
            bsdfs_to_remove.append(bsdf)
    
    print(f"Found {len(id_mappings)} material mappings")
    print(f"Will keep {len(unique_sionna_bsdfs)} unique Sionna materials")
    print(f"Will remove {len(bsdfs_to_remove)} duplicate BSDFs")
    
    # Replace original BSDFs with Sionna BSDFs (for the first occurrence of each)
    for new_id, info in unique_sionna_bsdfs.items():
        original = info['original_element']
        sionna_bsdf = info['element']
        parent = info['parent']
        index = info['index']
        
        # Replace the original element with the Sionna version
        parent.remove(original)
        parent.insert(index, sionna_bsdf)
        print(f"  - Created: {new_id} (replaces {original.get('id')}) -> {info['itu_material_name']}")
    
    # Remove duplicate BSDFs
    for duplicate in bsdfs_to_remove:
        parent = root
        parent.remove(duplicate)
        print(f"  - Removed duplicate: {duplicate.get('id')}")
    
    # Second pass: update all references in the file
    update_count = update_references(root, id_mappings)
    print(f"\nUpdated {update_count} references in shape elements")
    
    # Save the modified XML
    tree.write(output_file, encoding='UTF-8', xml_declaration=True)
    
    # Create pretty-printed version
    with open(output_file, 'r', encoding='utf-8') as f:
        xml_content = f.read()
    
    # Parse and pretty print
    parsed = minidom.parseString(xml_content)
    pretty_xml = parsed.toprettyxml(indent="  ")
    
    # Remove extra blank lines
    lines = [line for line in pretty_xml.split('\n') if line.strip() != '']
    pretty_xml = '\n'.join(lines)
    
    # Write the pretty-printed version
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(pretty_xml)
    
    print(f"\n✅ Saved processed XML to: {output_file}")
    
    return id_mappings, unique_sionna_bsdfs


def update_references(root, id_mappings):
    """
    Updates all references to old material IDs with the new Sionna IDs.
    Returns the number of references updated.
    """
    update_count = 0
    
    # Update all <ref> elements
    for ref in root.findall('.//ref'):
        old_id = ref.get('id')
        if old_id in id_mappings:
            new_id = id_mappings[old_id][0]
            ref.set('id', new_id)
            update_count += 1
    
    # Also check for any other elements that might reference materials by ID
    # This includes shape elements that might have material attributes
    for shape in root.findall('.//shape'):
        # Check for bsdf attributes
        for elem in shape:
            if elem.tag == 'ref' and elem.get('id') in id_mappings:
                new_id = id_mappings[elem.get('id')][0]
                elem.set('id', new_id)
                update_count += 1
    
    return update_count
  

# def standardize_xml(input_path, output_path):
#     try:
#         # Process the XML file
#         process_xml_file(input_path, output_path)
        
#         print("\n" + "="*60)
#         print("CONVERSION COMPLETE!")
#         print("="*60)
        
#     except FileNotFoundError:
#         print(f"❌ Error: Input file '{input_path}' not found.")
#         print("Please update the 'input_xml' variable in the script.")
#     except ET.ParseError as e:
#         print(f"❌ Error parsing XML file: {e}")
#     except Exception as e:
#         print(f"❌ Unexpected error: {e}")


# if __name__ == "__main__":
#     standardize_xml()
    