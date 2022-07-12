import os

from PyPDF2 import PdfReader

pdf = os.path.join("pdf", "Encyclopedia of Plants and Flowers.pdf") #example
reader = PdfReader(pdf)
with open("extracted_text.txt",'w',encoding='utf-8') as f:

    for i in range(60,496):
        page = reader.pages[i]
        # print(page.extract_text())
        f.write(page.extract_text())
f.close()