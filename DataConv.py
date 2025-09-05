import csv

input_path = r"D:\BITS Acad\4-1\nlp_assn_1\datasets\english\english_test.txt"
output_path = r"D:\BITS Acad\4-1\nlp_assn_1\datasets\english\english_test.csv"

with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", newline="", encoding="utf-8") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["genre", "description"])
    for line in infile:
        if line.strip():
            parts = line.strip().split(",", 1)
            if len(parts) == 2:
                genre = parts[0].strip().strip("'\"")
                desc = parts[1].strip().strip("'\"")
                writer.writerow([genre, desc])

print("Converted dataset saved to:", output_path)
