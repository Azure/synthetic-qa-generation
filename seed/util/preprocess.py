import os
import cv2
import fitz  # PyMuPDF
import base64
import langchain
import mimetypes
import requests
from collections import defaultdict

from PIL import Image
import numpy as np

from glob import glob
from markdownify import MarkdownConverter
from bs4 import BeautifulSoup
from langchain_text_splitters import CharacterTextSplitter
from PIL import Image
from mimetypes import guess_type

class ModifiedMarkdownConverter(MarkdownConverter):
    def convert_td(self, el, text, convert_as_inline):
        colspan = 1
        if "colspan" in el.attrs:
            try:
                colspan = int(el["colspan"])
            except ValueError:
                colspan = 1  # Default to 1 if conversion fails
        return " " + text.strip().replace("\n", " ") + " |" * colspan

    def convert_th(self, el, text, convert_as_inline):
        colspan = 1
        if "colspan" in el.attrs:
            try:
                colspan = int(el["colspan"])
            except ValueError:
                colspan = 1  # Default to 1 if conversion fails
        return " " + text.strip().replace("\n", " ") + " |" * colspan


def markdownify(
    html: str, **options
) -> str:
    return ModifiedMarkdownConverter(**options).convert(html)


def analyze_pdf_page_content(pdf_path, text_length_thres=600):
    document = fitz.open(pdf_path)
    page_analysis = defaultdict(list)

    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text = page.get_text("text")
        image_list = page.get_images(full=True)

        text_length = len(text)
        num_images = len(image_list)

        if text_length > text_length_thres and num_images == 0:
            content_type = 'Text'
        elif text_length <= text_length_thres and num_images > 0:
            content_type = 'Image'
        else:
            content_type = 'Mixed'            

        page_analysis[content_type].append(page_num)

    return dict(page_analysis)


def split_pdf(input_path, output_path, pages):
    pdf_document = fitz.open(input_path)
    output_pdf = fitz.open()
    
    for page_num in pages:
        output_pdf.insert_pdf(pdf_document, from_page=page_num, to_page=page_num)
    
    output_pdf.save(output_path)
    output_pdf.close()
    pdf_document.close()


def split_pdf_with_page_numbers(input_path, output_path, pages):
    pdf_document = fitz.open(input_path)
    output_pdf = fitz.open()
    
    for page_num in pages:
        page = pdf_document.load_page(page_num )
        # 원본 페이지 번호 추가
        page_text = "Original Page: {}".format(page_num)
        rect = page.rect
        point = fitz.Point(rect.width - 100, rect.height - 20)
        page.insert_text(point, page_text, fontsize=12, color=(0, 0, 0))
        output_pdf.insert_pdf(pdf_document, from_page=page_num, to_page=page_num)
    
    output_pdf.save(output_path)
    output_pdf.close()
    pdf_document.close()


def crop_image_from_image(image_path, page_number, bounding_box):
    """
    Crops an image based on a bounding box.

    :param image_path: Path to the image file.
    :param page_number: The page number of the image to crop (for TIFF format).
    :param bounding_box: A tuple of (left, upper, right, lower) coordinates for the bounding box.
    :return: A cropped image.
    :rtype: PIL.Image.Image
    """
    with Image.open(image_path) as img:
        if img.format == "TIFF":
            # Open the TIFF image
            img.seek(page_number)
            img = img.copy()
            
        # The bounding box is expected to be in the format (left, upper, right, lower).
        cropped_image = img.crop(bounding_box)
        return cropped_image


def crop_image_from_pdf_page(pdf_path, page_number, bounding_box):
    """
    Crops a region from a given page in a PDF and returns it as an image.

    :param pdf_path: Path to the PDF file.
    :param page_number: The page number to crop from (0-indexed).
    :param bounding_box: A tuple of (x0, y0, x1, y1) coordinates for the bounding box.
    :return: A PIL Image of the cropped area.
    """
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number)
    
    # Cropping the page. The rect requires the coordinates in the format (x0, y0, x1, y1).
    bbx = [x * 72 for x in bounding_box]
    rect = fitz.Rect(bbx)
    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72), clip=rect)
    
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    doc.close()

    return img


def crop_image_from_file(file_path, page_number, bounding_box):
    """
    Crop an image from a file.

    Args:
        file_path (str): The path to the file.
        page_number (int): The page number (for PDF and TIFF files, 0-indexed).
        bounding_box (tuple): The bounding box coordinates in the format (x0, y0, x1, y1).

    Returns:
        A PIL Image of the cropped area.
    """
    mime_type = mimetypes.guess_type(file_path)[0]
    
    if mime_type == "application/pdf":
        return crop_image_from_pdf_page(file_path, page_number, bounding_box)
    else:
        return crop_image_from_image(file_path, page_number, bounding_box)
    

# Function to encode a local image into data URL 
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"


def update_figure_description(md_content, img_description, idx):
    """
    Updates the figure description in the Markdown content.

    Args:
        md_content (str): The original Markdown content.
        img_description (str): The new description for the image.
        idx (int): The index of the figure.

    Returns:
        str: The updated Markdown content with the new figure description.
    """    
    # The substring you're looking for    
    start_substring = f"![](figures/{idx})"
    end_substring = "</figure>"
    new_string = f"<!-- FigureContent=\"{img_description}\" -->"

    grouped_content_start_substring = "<figure>"
    new_md_content = md_content
    # Find the start and end indices of the part to replace
    start_index = md_content.find(start_substring)
    
    if start_index != -1:  # if start_substring is found
        start_index += len(start_substring)  # move the index to the end of start_substring
        end_index = md_content.find(end_substring, start_index)
        if end_index != -1:  # if end_substring is found
            # Replace the old string with the new string
            new_md_content = md_content[:start_index] + new_string + md_content[end_index:]

    group_content_index = md_content.find(grouped_content_start_substring)
    if group_content_index != -1:  # if start_substring is found
        start_index += len(grouped_content_start_substring)  # move the index to the end of start_substring
        end_index = md_content.find(end_substring, start_index)
        if end_index != -1:  # if end_substring is found
            # Replace the old string with the new string
            new_md_content = md_content[:end_index] + "Image description :\n" + img_description + md_content[end_index:]

    return new_md_content


def understand_image_with_gpt(client, deployment_name, image_path, caption="", max_tokens=1024, language="Korean"):

    data_url = local_image_to_data_url(image_path)
    if caption == "":
        prompt = f"Describe this image in {language} language. " 
    else: 
        prompt = f"Describe this image in {language} language (note: it has image caption: {caption})."

    response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                { "role": "system", "content": "You are a helpful assistant." },
                { "role": "user", "content": [  
                    { 
                        "type": "text", 
                        "text": prompt
                    },
                    { 
                        "type": "image_url",
                        "image_url": {
                            "url": data_url
                        }
                    }
                ] } 
            ],
            max_tokens=max_tokens
        )

    img_description = response.choices[0].message.content
    
    return img_description    


def is_bounding_box_larger_than(bbox, min_width=1, min_height=1.0):
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0
    print(width, height)
    return width > min_width and height > min_height


# Function to calculate complexity using variance of Laplacian and Canny edge detection
def image_complexity(img, laplacian_var_thres=500, edge_count_thres=10000, total_entropy_thres=5.0):
    
    if isinstance(img, Image.Image):
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    ##### Histogram entropy
    hist_b = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([img], [2], None, [256], [0, 256])
    
    # Normalize the histograms
    hist_b /= hist_b.sum()
    hist_g /= hist_g.sum()
    hist_r /= hist_r.sum()
    
    # Calculate histogram entropy
    entropy_b = -np.sum(hist_b * np.log2(hist_b + 1e-7))
    entropy_g = -np.sum(hist_g * np.log2(hist_g + 1e-7))
    entropy_r = -np.sum(hist_r * np.log2(hist_r + 1e-7))
    
    # Total entropy
    total_entropy = entropy_b + entropy_g + entropy_r

    ### Laplacian variance
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray_img, cv2.CV_64F).var()
    
    ### Canny edge detection
    edges = cv2.Canny(gray_img, 100, 200)
    edge_count = np.sum(edges > 0)

    if laplacian_var > laplacian_var_thres or edge_count > edge_count_thres or total_entropy > total_entropy_thres:
        return "Complex", laplacian_var, edge_count, total_entropy
    else:
        return "Simple", laplacian_var, edge_count, total_entropy
    

def clean_html(html):  
    try:
        # Parse the HTML content
        soup = BeautifulSoup(html, 'html.parser')

        # Remove problematic attributes that might cause conversion issues
        for tag in soup.find_all():
            # List of attributes known to cause issues
            problematic_attributes = ['colspan', 'rowspan', 'width', 'height']
            for attr in problematic_attributes:
                if attr in tag.attrs:
                    del tag[attr]  # Removing the attribute altogether

        # Using a simpler method to convert HTML to text, avoiding markdownify issues
        text = soup.get_text(separator=' ', strip=True)
        # Replace multiple spaces with a single space
        text = ' '.join(text.split())
    except Exception as e:
        print(f"Error processing HTML: {e}\nHTML segment causing issue: {html[:500]}")
        # Use raw text extraction as a fallback
        text = BeautifulSoup(html, 'html.parser').get_text(separator=' ', strip=True)
        text = ' '.join(text.split())
    
    return text


def convert_html_to_md(html):
    # Convert html to markdown
    try:
        text = markdownify(html)    
    except Exception as e:
        text = clean_html(html)
    return text


def remove_small_images(image_path, image_dim_thres=160000):
    images_ = glob(os.path.join(image_path, "*"))
    images = []
    for image in images_:
        img = cv2.imread(image)
        h, w, _ = img.shape
        if h*w < 160000: # 
            os.remove(image)
        else:
            images.append(image)
    return images


def remove_short_sentences(text_summaries, thres=10):
    remove_indices = []
    for idx, text in enumerate(text_summaries):
        num_words = len(text.split())
        if num_words < thres:
            remove_indices.append(idx)

    text_summaries = [i for j, i in enumerate(text_summaries) if j not in remove_indices]
    return text_summaries


def encode_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    

def encode_url_image_base64(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        encoded_image = base64.b64encode(response.content).decode("utf-8")
    else:
        encoded_image = None
    return encoded_image


def split_text_using_tiktoken(texts, chunk_size, chunk_overlap, encoding_name="o200k_base"):
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n\n", 
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        encoding_name=encoding_name
    )

    if isinstance(texts[0], langchain.schema.Document):
        a = [text.page_content for text in texts]
    else:
        a = [text for text in texts]
    joined_texts = '\n\n'.join(a)
    texts_tiktoken = text_splitter.split_text(joined_texts)
    
    return texts_tiktoken