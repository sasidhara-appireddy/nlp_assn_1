import csv
import re
#Copy the path and just replace .txt with .csv at the end 
input_file = r"D:\BITS Acad\4-1\nlp_assn_1\datasets\hindi\hindi_test.txt"
output_file = r"D:\BITS Acad\4-1\nlp_assn_1\datasets\hindi\hindi_test.csv"

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", newline="", encoding="utf-8") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["text", "labels"])

    for line in infile:
        line = line.strip()
        if not line:
            continue  

        parts = re.findall(r"'(.*?)'", line)

        if len(parts) < 2:
            continue  

        categories = parts[0].strip().strip("[]")
        text = parts[1].strip()

        labels = [c.strip() for c in categories.split(",")]

        writer.writerow([text, ",".join(labels)])
