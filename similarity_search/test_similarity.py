from fuzzywuzzy import fuzz, process

# Your data
ocr_text = "H0u6Ang-C7RL-I^t"  # OCR output with error

# Different fuzzy matching algorithms
print("1. SIMPLE RATIO:")
ratio = fuzz.ratio(ocr_text, "Tampines")
print(f"   fuzz.ratio('{ocr_text}', 'Tampines') = {ratio}%")
print(f"   → Measures overall similarity character by character")
print()

print("2. PARTIAL RATIO:")
partial = fuzz.partial_ratio(ocr_text, "Tampines")
print(f"   fuzz.partial_ratio('{ocr_text}', 'Tampines') = {partial}%")
print(f"   → Finds best matching substring")
print()

print("3. TOKEN SORT RATIO:")
token_sort = fuzz.token_sort_ratio(ocr_text, "Tampines")
print(f"   fuzz.token_sort_ratio('{ocr_text}', 'Tampines') = {token_sort}%")
print(f"   → Sorts words alphabetically then compares")
print()

print("4. TOKEN SET RATIO:")
token_set = fuzz.token_set_ratio(ocr_text, "Tampines")
print(f"   fuzz.token_set_ratio('{ocr_text}', 'Tampines') = {token_set}%")
print(f"   → Compares unique words, ignoring duplicates")
print()

# Demonstrate with multiple potential destinations
multiple_destinations = ["Houngang Ave Int",
                         "HOUGANG CTRL INT", "Houngang Ctrl Int"]
all_matches = process.extract(ocr_text, multiple_destinations)

print(f"   Comparing '{ocr_text}' against: {multiple_destinations}")
print("   Results:")
for dest, score in all_matches:
    print(f"   - '{dest}': {score}%")

best_from_multiple = process.extractOne(ocr_text, multiple_destinations)
print(
    f"\n   → Best match: '{best_from_multiple[0]}' ({best_from_multiple[1]}%)")


print("a-C7RL-I^t temu".capitalize())