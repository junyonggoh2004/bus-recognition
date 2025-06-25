
import pandas as pd
import re
import string
from fuzzywuzzy import process

import pandas as pd
from fuzzywuzzy import process

class SimilaritySearch:
    """
    A class to extract origin and destination for a given bus route number
    and perform fuzzy matching with OCR-extracted text.
    """

    def __init__(self):
        """
        Initializes the SimilaritySearch with a fixed file path.
        """
        self.file_path = "bus_information.csv"
        self.bus_number = None

    def extract_wiki_origin_and_destination(self, bus_number):
        """
        Extract origin and destination for a given bus number from Excel file.

        Args:
            bus_number (str): Bus number to search for

        Returns:
            list: List containing [origin, destination], with '-' for blank or missing values.
        """
        self.bus_number = bus_number

        try:
            df = pd.read_csv(self.file_path)
            rows = df[df['Route number'] == bus_number.upper()]

            if rows.empty:
                return ['Route not found', 'Route not found']

            result_list = []

            # Iterate through all matching rows
            for _, row in rows.iterrows():
                origin = row['Origin'] if pd.notna(row['Origin']) else '-'
                destination = row['Destination'] if pd.notna(
                    row['Destination']) else '-'

                # Add both origin and destination to the result list
                result_list.append(origin)
                result_list.append(destination)

            return result_list

        except Exception as e:
            return [f"Error: {str(e)}", f"Error: {str(e)}"]

    def match_ocr_text(self, ocr_text, locations):
        """
        Perform fuzzy matching between OCR-extracted text and a list of location names.

        Args:
            ocr_text (str): Text extracted via OCR that needs to be matched.
            location_pair (list): A list containing two location strings [origin, destination].

        Returns:
            None: Prints comparison results and best match.
        """
        all_matches = process.extract(ocr_text, locations)

        print(f"\nComparing OCR text '{ocr_text}' against: {locations}")
        print("Results:")
        for dest, score in all_matches:
            print(f"   - '{dest}': {score}%")

        best_match = process.extractOne(ocr_text, locations)
        print(f"\n→ Best match: '{best_match[0]}' ({best_match[1]}%)")
        return self.bus_number, best_match[0]

# 45: Upp3 E Coa5t Bu5 TeT Ang Mo Ki0 Stret 63
# 45A: L0r Ah 500 (0pp B1k 115) Serang00n MRT statin
# 46: Uppr E C0st Bus Teminal Pasir Ris Buz Interchnge
# 47:Chn9i 8iz Pk BuS T3R Amb3r Rod (lo0p)
# 48: B3d0k Nt Bu5 D3p07 Bu0na Vsta Bus Termnal
