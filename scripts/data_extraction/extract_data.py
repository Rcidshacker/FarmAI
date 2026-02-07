import os
import csv
import datetime
import re
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

def _get_exif_data(image_path):
    """
    Extracts relevant EXIF data from a single image file.
    """
    exif_data = {}
    try:
        image = Image.open(image_path)
        info = image.getexif()

        if not info:
            return {
                'EXIF_DateTime': None,
                'EXIF_Camera_Make': None,
                'EXIF_Camera_Model': None,
                'EXIF_Latitude': None,
                'EXIF_Longitude': None
            }

        # Decode all EXIF tags
        for tag_id, value in info.items():
            tag = TAGS.get(tag_id, tag_id)
            exif_data[tag] = value

        # --- 1. Get DateTime ---
        dt = exif_data.get('DateTimeOriginal') or exif_data.get('DateTime')
        
        # --- 2. Get Camera Info ---
        make = exif_data.get('Make')
        model = exif_data.get('Model')

        # --- 3. Get GPS Info ---
        latitude = None
        longitude = None
        gps_info = exif_data.get('GPSInfo')
        
        if gps_info:
            lat_data = gps_info.get(2) # Latitude
            lat_ref = gps_info.get(1)  # N/S
            lon_data = gps_info.get(4) # Longitude
            lon_ref = gps_info.get(3)  # E/W

            if lat_data and lat_ref and lon_data and lon_ref:
                latitude = _convert_dms_to_dd(lat_data, lat_ref)
                longitude = _convert_dms_to_dd(lon_data, lon_ref)

        return {
            'EXIF_DateTime': dt,
            'EXIF_Camera_Make': make.strip() if make else None,
            'EXIF_Camera_Model': model.strip() if model else None,
            'EXIF_Latitude': latitude,
            'EXIF_Longitude': longitude
        }

    except Exception as e:
        print(f"  [!] Error processing EXIF for {os.path.basename(image_path)}: {e}")
        return {
            'EXIF_DateTime': None,
            'EXIF_Camera_Make': None,
            'EXIF_Camera_Model': None,
            'EXIF_Latitude': None,
            'EXIF_Longitude': None
        }

def _convert_dms_to_dd(dms, ref):
    """
    Converts (Degrees, Minutes, Seconds) and reference (N/S/E/W) to Decimal Degrees.
    """
    try:
        degrees = dms[0]
        minutes = dms[1]
        seconds = dms[2]
        
        dd = float(degrees) + float(minutes)/60 + float(seconds)/3600
        
        if ref in ('S', 'W'):
            dd = -dd
        return dd
    except:
        return None

def parse_filename(filename):
    """
    Parses the filename based on the observed patterns like:
    'Ruchit_3_lower_flower_1762070087546.jpeg'
    'siddharth2_1_lower_flower_1762075068981.jpeg'
    
    Returns a dictionary of the parts.
    """
    parts = os.path.splitext(filename)[0].split('_')
    
    # Pattern 1: 'Ruchit_3_lower_flower_...'
    if len(parts) == 5:
        return {
            'Filename_Collector': parts[0],
            'Plant_ID': parts[1],
            'Position': parts[2],
            'Part_Observed': parts[3],
            'Filename_Timestamp_ms': parts[4],
            'Parsing_Pattern': 'Pattern 1 (5 parts)'
        }
    # Pattern 2: 'siddharth2_1_lower_flower_...'
    elif len(parts) == 6:
         return {
            'Filename_Collector': parts[0],
            'Plant_ID': f"{parts[1]}_{parts[2]}", # e.g., '2_1'
            'Position': parts[3],
            'Part_Observed': parts[4],
            'Filename_Timestamp_ms': parts[5],
            'Parsing_Pattern': 'Pattern 2 (6 parts)'
        }
    else:
        # Fallback for unrecognized patterns
        return {
            'Filename_Collector': None,
            'Plant_ID': None,
            'Position': None,
            'Part_Observed': None,
            'Filename_Timestamp_ms': None,
            'Parsing_Pattern': 'Unknown'
        }

def process_all_images(base_directory, output_csv_file):
    """
    Main function to walk all subdirectories and process images.
    """
    
    # These are the columns for your final CSV
    csv_headers = [
        'Path_Collector',
        'Full_Path',
        'Filename',
        'Filename_Collector',
        'Plant_ID',
        'Position',
        'Part_Observed',
        'Filename_Timestamp_ms',
        'Filename_DateTime',
        'EXIF_DateTime',
        'EXIF_Camera_Make',
        'EXIF_Camera_Model',
        'EXIF_Latitude',
        'EXIF_Longitude',
        'Parsing_Pattern'
    ]
    
    all_image_data = []
    
    # We use os.walk to go through every folder and file
    for root, dirs, files in os.walk(base_directory):
        # We only care about folders inside the base directory
        # e.g., '.../Custurd Apple Images/ruchit'
        relative_path = os.path.relpath(root, base_directory)
        path_parts = relative_path.split(os.sep)
        
        # The first folder name is the collector (e.g., 'ruchit', 'siddharth')
        if path_parts and path_parts[0] != '.':
            path_collector = path_parts[0]
        else:
            continue # Skip files in the root folder itself

        print(f"--- Processing folder: {path_collector} ---")

        for filename in files:
            file_lower = filename.lower()
            if file_lower.endswith(('.jpg', '.jpeg', '.png', '.tiff')):
                print(f"  -> Processing file: {filename}")
                full_path = os.path.join(root, filename)
                
                # 1. Get data from filename
                file_info = parse_filename(filename)
                
                # 2. Convert timestamp from filename
                dt_from_filename = None
                if file_info['Filename_Timestamp_ms']:
                    try:
                        ts_sec = int(file_info['Filename_Timestamp_ms']) / 1000
                        dt_from_filename = datetime.datetime.fromtimestamp(ts_sec).strftime('%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        pass # Timestamp was invalid

                # 3. Get data from EXIF metadata
                exif_info = _get_exif_data(full_path)
                
                # 4. Combine all data into one row
                row_data = {
                    'Path_Collector': path_collector,
                    'Full_Path': full_path,
                    'Filename': filename,
                    'Filename_Collector': file_info['Filename_Collector'],
                    'Plant_ID': file_info['Plant_ID'],
                    'Position': file_info['Position'],
                    'Part_Observed': file_info['Part_Observed'],
                    'Filename_Timestamp_ms': file_info['Filename_Timestamp_ms'],
                    'Filename_DateTime': dt_from_filename,
                    'EXIF_DateTime': exif_info['EXIF_DateTime'],
                    'EXIF_Camera_Make': exif_info['EXIF_Camera_Make'],
                    'EXIF_Camera_Model': exif_info['EXIF_Camera_Model'],
                    'EXIF_Latitude': exif_info['EXIF_Latitude'],
                    'EXIF_Longitude': exif_info['EXIF_Longitude'],
                    'Parsing_Pattern': file_info['Parsing_Pattern']
                }
                all_image_data.append(row_data)

    # 5. Write all collected data to the CSV file
    with open(output_csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(all_image_data)

    print(f"\n✅ Success! All data extracted to '{output_csv_file}'")


# ---
# ⚠️ 1. CHANGE THIS to your main dataset folder
# ---
# Use raw string (r'...') on Windows to handle backslashes
BASE_DIR = r'C:\Users\Ruchit\Desktop\Code\2025\eIPL\Dataset\Custurd Apple Images-20251105T045712Z-1-001\Custurd Apple Images'

# ---
# ⚠️ 2. CHANGE THIS to where you want to save the final CSV
# ---
OUTPUT_CSV = r'C:\Users\Ruchit\Desktop\Code\2025\eIPL\New custard\custardcode\extracted_data.csv'


# ---
# 3. Run the script
# ---
if __name__ == "__main__":
    if not os.path.isdir(BASE_DIR):
        print(f"Error: Base directory not found: {BASE_DIR}")
    else:
        process_all_images(BASE_DIR, OUTPUT_CSV)