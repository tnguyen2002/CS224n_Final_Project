import PyPDF2
import os
import glob
import random
import re
# import pytesseract
# from PIL import Image
# from pdf2image import convert_from_path
# mv */*_extracted.txt /data/reports

def remove_nonalphanumeric(string):
    # pattern = r'\w+' # Matches one or more alphanumeric characters
    output_str = re.sub(r'\W+', ' ', string)
    return output_str

def extractDiagnosis(txt_path,file_name):
    # Specify the starting tokens and number of words to extract
    print("text path = ", txt_path)
    start_tokens = ['Diagnosis', 'diagnosis', "DIAGNOSIS"]
    num_words = 200

    # Open the text file and read its contents
    with open(txt_path, 'r') as f:
        text = f.read()

    # Find the starting index of the first matching token
    start_index = -1
    for start_token in start_tokens:
        index = text.find(start_token)
        if index != -1:
            start_index = index
            break

    # If no matching token was found, exit the program
    if start_index == -1:

        print(f"No matching start token found in file {txt_path}")
        start_index = random.randint(0, len(text) - 200)
        start_index = max(start_index, 0)

        # exit()

    # Find the end index of the section
    end_index = start_index
    for i in range(min(num_words,len(text))):
        end_index = text.find(' ', end_index + 1)
        if end_index == -1:
            end_index = len(text)
            break
            
    # Extract the section of text
    section_text = text[start_index:end_index]
    section_text = section_text.replace('\n', '')
    section_text = remove_nonalphanumeric(section_text)
    file_path = "../../../../../data/data/reports"
    file_path = os.path.join(file_path, file_name + "_extracted" + '.txt')
    print(file_path)
    with open(file_path, 'w') as f:
        f.write(section_text)

    # Print a confirmation message
    print(f"Section extracted and saved to {file_path}")

directory = "../../../../../../data/data/pdfs/"
unextractedDirectory = "../../../../../../data/data/reports_unextracted"
# Get a list of all the files and folders in the directory
items = os.listdir(directory)
# Filter the list to only include directories
folders = [item for item in items if os.path.isdir(os.path.join(directory, item))]
folders.sort()
Error_Count = 0
# Use glob to find all files with the .pdf extension in the directory
pdf_files = glob.glob(os.path.join(directory, "*.PDF"))
# print(pdf_files)
for pdf_file in pdf_files:
# for i in range(1):
    pdf_file_path = pdf_file
    pdf_file = pdf_file_path.split("/")[-1][:-4]
    print(pdf_file_path)
    print(pdf_file)
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file_path)
        # Read the text from each page
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        unextracted_path = os.path.join(unextractedDirectory, pdf_file + '.txt')
        # print(unextracted_path)
        with open(unextracted_path, 'w') as f:
            f.write(text)
        extractDiagnosis(txt_path=unextracted_path, file_name=pdf_file)
    except:
        Error_Count += 1
print(Error_Count)
