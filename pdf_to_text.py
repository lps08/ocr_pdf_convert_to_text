from unicodedata import name
import cv2
import pytesseract
from pdf2image import convert_from_path
import numpy as np
import matplotlib.pyplot as plt

class PDF2Text(object):
    def __init__(self, pdf_path:str, ocr_config:str = '--psm 1', num_pages:int=1) -> None:
        self.ocr_config = ocr_config
        self.pdf_path = pdf_path
        self.image_quality = 500
        self.num_pages = num_pages

    def get_pages(self):
        preprocessed_pages = []
        pages = convert_from_path(self.pdf_path, self.image_quality, first_page=1, last_page=self.num_pages, thread_count=4, grayscale=True)
        for page in pages:
            page = np.uint8(page)
            # removing noise
            page = cv2.medianBlur(page , 5)
            # applying otsu threshold
            page = cv2.threshold(page, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

            preprocessed_pages.append(page)

        return preprocessed_pages

    def __str__(self) -> str:
        pages = self.get_pages()
        texts = [pytesseract.image_to_string(page, config=self.ocr_config, lang='por') for page in pages]
        return '\n\n'.join(texts)

if __name__ == '__main__':
    pdf_test = PDF2Text(pdf_path='doc.pdf', num_pages=1)
    cv2.imshow('page', pdf_test.get_pages()[0])
    print(pdf_test)