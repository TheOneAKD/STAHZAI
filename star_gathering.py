import pandas as pd
import requests
from astroquery.simbad import Simbad
from astroquery.skyview import SkyView
from astroquery.vizier import Vizier
from io import BytesIO
from PIL import Image
import os
from astropy.io import fits
import matplotlib.pyplot as plt

# Define the directory to save images
image_dir = r"C:\Users\Jyoti\OneDrive\Desktop\Coding\SciRe 2024-25 STAHZAI\images"
os.makedirs(image_dir, exist_ok=True)  # Create directory if it doesn't exist
badNameCounter = 0

# Define a custom Simbad query to get specific details about stars
Simbad.add_votable_fields('flux(V)', 'otype', 'sptype', 'distance', 'diameter')

def fetch_star_image(star_name):
    try:
        # Query SkyView for images centered on the star
        images = SkyView.get_images(position=star_name, survey='DSS', pixels='300,300')
        if images:
            fits_data = images[0][0]
            img_name = os.path.join(image_dir, f"{star_name.replace(' ', '_')}.jpg")
            
            # Open the FITS file and plot it
            data = fits_data.data
            plt.figure()
            plt.imshow(data, cmap='gray', origin='lower')
            plt.axis('off')
            plt.savefig(img_name, bbox_inches='tight', pad_inches=0, dpi=150)
            plt.close()
            
            print(f"Image saved as {img_name}")
            return img_name
        else:
            print(f"No image found for {star_name}")
            return None
    except Exception as e:
        print(f"Failed to fetch image for {star_name}: {e}")
        return None

def query_star_info(star_name):
    if not star_name.strip():
        print(f"Skipping invalid star name: {star_name}")
        return None
    
    result = Simbad.query_object(star_name)
    # print(result.colnames)
    if result:
        star_type = result['SP_TYPE'][0] if 'SP_TYPE' in result.colnames else "Unknown"
        luminosity = result['FLUX_V'][0] if 'FLUX_V' in result.colnames and result['FLUX_V'][0] else "Unknown"
        diameter = result['Diameter_diameter'][0] if 'Diameter_diameter' in result.colnames else "Unknown"
        composition = result['COMP_ELEM'][0] if 'COMP_ELEM' in result.colnames else "Unknown"
        
        # # Attempt to fetch real composition data from available fields
        # composition = "Unknown"
        # if 'fe_h' in result.colnames and result['fe_h'][0]:
        #     composition = f"Metallicity: [Fe/H] = {result['fe_h'][0]}"
        
        color = "Unknown"  # Color can be deduced or calculated based on spectral type
        
        return {
            "Star Name": star_name,
            "Size (in miles)": diameter,
            "Type (OBAFGKM Scale)": star_type,
            "Luminosity (Absolute Magnitude)": luminosity,
            "Color": color,
            "Composition": composition # CANNOT YET BE FOUND
        }
    else:
        print(f"Could not retrieve information for {star_name}.")
        return None

def get_star_names(required_count, badNameCounter):
    # Retrieve a list of known star names from Vizier's Bright Star Catalogue
    Vizier.ROW_LIMIT = 1000  # Temporarily set a high limit to get more names if needed
    result = Vizier.get_catalogs("V/50")[0]  # Bright Star Catalogue
    star_names = result['Name']
    
    # Clean up and filter names
    clean_names = []
    for name in star_names:
        # print(name)
        if len(clean_names) >= required_count:
            break  # Stop once we have enough valid names
        
        clean_name = str(name).strip()
        # Check if Simbad recognizes the name
        if clean_name and Simbad.query_object(clean_name):
            clean_names.append(clean_name)
        else:
            # print(f"Skipping invalid or unrecognized star name: {clean_name}")
            # print(badNameCounter)
            badNameCounter += 1
    
    return clean_names, badNameCounter

def compile_dataset(row_count):
    # Use real star names retrieved automatically
    star_names, counter = get_star_names(row_count, badNameCounter)
    data = []
    
    for star_name in star_names:
        star_info = query_star_info(star_name)
        if star_info:
            image_path = fetch_star_image(star_name)
            star_info["Image"] = image_path if image_path else "No Image Available"
            data.append(star_info)

    # Create DataFrame
    df = pd.DataFrame(data)
    # Save to CSV
    output_file = "star_dataset.csv"
    df.to_csv(output_file, index=False)
    print(f"Dataset compiled and saved as {output_file}")
    print(f"Bad Name Counter: {counter}")

# Example usage:
row_count = 100  # Number of real star names to retrieve and use
compile_dataset(row_count)
