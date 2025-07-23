import json
import re

import fitz

from entity_graph.config import setup_logger
from entity_graph.graph_extractor.processing.extraction_models import (
    PageComponents,
    Span,
)
from entity_graph.graph_extractor.processing.models import (
    TextExtracionConfig,
    TxtExtracionConfig,
)

logger = setup_logger(__name__)

"""
Contains:
extracting page
listing blocks in a page (with grouping by distance threshold)
finding blocks near given text, with filtering by y-overlap and right-to condition
groups blocks by y-axis distance to find split texts (NOTE: needs x-axis condition on distance)
NOTE: currently distance threshold is large enough to find split spans in one go,
    may be better to find the first and look for more with lower threshold
"""


def read_txt(config: TxtExtracionConfig) -> str | dict:
    """
    Reads a file as text.
    If it passes json parsing, returns  json dict.
    """
    filename = config.filename

    with open(filename, "r") as f:
        data = f.read()

    try:
        content = json.loads(data)
    except Exception as e:
        logger.info(f"Text file is not json {e}")
    else:
        data = content

    return data


def read_page(
    input_pdf_path,
    page_number: int | None = None,
    debug: bool = False,
) -> PageComponents:
    """
    Lists text blocks from a PDF along with their font size and color.
    Only return blocks within `threshold` of the input text `label`.

    :param input_pdf_path: Path to the input PDF file.
    :param page_number: Optional page number.
    """
    try:
        # Open the input PDF file
        pdf_document = fitz.open(input_pdf_path)

        pages = (
            [page_number - 1]
            if page_number is not None
            else range(pdf_document.page_count)
        )

        block_texts = []
        block_positions = []
        for page_number in pages:
            page = pdf_document.load_page(page_number)
            if debug:
                print(f"Page {page_number + 1}:")

            # Extract text blocks with their styles
            for block in page.get_text("dict")["blocks"]:
                texts = [
                    (span["text"], span["bbox"][0])
                    for line in block["lines"]
                    for span in line["spans"]
                ]
                texts = list(map(lambda x: x[0], sorted(texts, key=lambda x: x[1])))
                block_texts.append("\t".join(texts))
                block_positions.append(block["bbox"])

        pdf_document.close()
    except Exception as e:
        raise
    return PageComponents(text_blocks=block_texts, text_positions=block_positions)


def distance_right_end(pos1: tuple, pos2: tuple) -> float:
    return ((pos1[2] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5


def find_right_side_y_axis_overlaps(
    reference_box: tuple, boxes: list, box_map=None
) -> list:
    """
    Find boxes that are to the right of the reference box and overlap on the y-axis.

    Args:
        reference_box (tuple): A tuple (x1, y1, x2, y2) defining the reference box.
        boxes (list): A list of tuples, where each tuple is (x1, y1, x2, y2) defining a box.

    Returns:
        list: A list of boxes that are to the right and overlap on the y-axis with the reference box.
    """
    x1_ref, y1_ref, x2_ref, y2_ref = reference_box

    overlapping_boxes = []
    for box in boxes:
        if box_map is None:
            x1, y1, x2, y2 = box
        else:
            x1, y1, x2, y2 = box_map(box)
        # Check if the y-axis ranges overlap
        y_overlap = max(y1_ref, y1) <= min(y2_ref, y2)
        # Check if the box is to the right
        is_to_the_right = x1 > x2_ref

        if y_overlap and is_to_the_right:
            overlapping_boxes.append(box)

    return overlapping_boxes


def group_boxes_by_y(
    boxes: list[Span], threshold: int, box_map=None
) -> list[list[Span]]:
    """
    Groups boxes if their y-axis distance is below the threshold.

    Args:
        boxes (List[Tuple[int, int, int, int]]): List of boxes, each defined as (x_min, y_min, x_max, y_max).
        threshold (int): Maximum y-axis distance for grouping.

    Returns:
        List[List[Tuple[int, int, int, int]]]: Groups of boxes satisfying the proximity condition.
    """

    def is_close_y(
        box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]
    ) -> bool:
        """Checks if two boxes are close on the y-axis."""
        if box_map is None:
            ...
        else:
            box1 = box_map(box1)
            box2 = box_map(box2)
        return (
            abs(box1[1] - box2[1]) <= threshold or abs(box1[3] - box2[3]) <= threshold
        )

    groups = []
    visited = set()

    for i, box in enumerate(boxes):
        if i in visited:
            continue

        # Create a new group
        group = [box]
        visited.add(i)

        for j, other_box in enumerate(boxes):
            if j not in visited and any(is_close_y(b, other_box) for b in group):
                group.append(other_box)
                visited.add(j)

        groups.append(group)

    groups = [sorted(g, key=lambda x: x.bbox[0]) for g in groups]  # by x0 in group
    groups = sorted(
        groups, key=lambda x: min(e.bbox[1] for e in x)
    )  # by y0 between groups
    return groups


def _find_label_info(
    input_pdf_path,
    page_number: int | None = None,
    label: str = "",
    threshold: float = 100,
    debug: bool = False,
    with_distance: bool = False,
) -> tuple[tuple, list[Span]] | None:
    """
    Lists text blocks from a PDF along with their font size and color.
    Only return blocks withing `threshold` of the input text `label`.

    :param input_pdf_path: Path to the input PDF file.
    :param page_number: Optional page number.
    """
    try:
        # Open the input PDF file
        pdf_document = fitz.open(input_pdf_path)

        pages = (
            [page_number] if page_number is not None else range(pdf_document.page_count)
        )

        target_block = None
        for page_number in pages:
            page = pdf_document.load_page(page_number)
            if debug:
                print(f"Page {page_number + 1}:")

            # Extract text blocks with their styles
            for block in page.get_text("dict")["blocks"]:
                if "lines" in block:
                    block_position = block[
                        "bbox"
                    ]  # Position of the block (x0, y0, x1, y1)
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"]
                            font_size = span["size"]
                            color = span["color"]
                            if label in text:
                                if debug:
                                    print(
                                        f"Text: {text}, Font Size: {font_size}, Color: #{color:06x}, Position: {block_position}"
                                    )
                                target_block = block

        pdf_document.close()
        if target_block is None:
            print(f"target block not found!")
            return

        # # Open the input PDF file
        pdf_document = fitz.open(input_pdf_path)
        pages = (
            [page_number] if page_number is not None else range(pdf_document.page_count)
        )
        # # if right_of
        target_pos = (target_block["bbox"][2], target_block["bbox"][1])

        close = []
        for page_number in pages:
            page = pdf_document.load_page(page_number)
            if debug:
                print(f"Page {page_number + 1}:")

            # Extract text blocks with their styles
            for block in page.get_text("dict")["blocks"]:
                if "lines" in block:
                    block_position = block[
                        "bbox"
                    ]  # Position of the block (x0, y0, x1, y1)
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"]
                            font_size = span["size"]
                            color = span["color"]
                            if (
                                distance_right_end(target_block["bbox"], span["bbox"])
                                < threshold
                            ):
                                if debug:
                                    print(
                                        f"Text: {text}, Font Size: {font_size}, Color: #{color:06x}, Position: {block_position}"
                                    )
                                if with_distance:
                                    close.append(
                                        (
                                            text,
                                            span["bbox"],
                                            distance_right_end(
                                                target_block["bbox"], span["bbox"]
                                            ),
                                        )
                                    )
                                else:
                                    close.append(Span.model_validate(span))

        pdf_document.close()
        return target_block["bbox"], close

    except Exception as e:
        print(f"An error occurred: {e}")


def find_label_info(
    input_pdf_path, page_number: int | None = None, label: str = ""
) -> list[str]:
    """
    Lists text blocks from a PDF along with their font size and color.
    Only return blocks within `threshold` of the input text `label`.
    Filters only blocks with y overlap and right-to target.
    Joins boxes with similar y and returns the text.

    :param input_pdf_path: Path to the input PDF file.
    :param page_number: Optional page number.
    """
    target_block, close = _find_label_info(input_pdf_path, page_number, label)
    r = find_right_side_y_axis_overlaps(target_block, close, box_map=lambda x: x.bbox)
    groups = group_boxes_by_y(r, 0.01, box_map=lambda x: x.bbox)
    return [" ".join(e.text for e in g) for g in groups]


def find_pattern(
    input_pdf_path,
    pattern: str = "",
    page_number: int | None = None,
    debug: bool = False,
) -> list[str]:
    """
    For a given pattern returns all matching text fields.
    Page `page_number` counting from 0.
    """

    try:
        # Open the input PDF file
        pdf_document = fitz.open(input_pdf_path)

        pages = (
            [page_number] if page_number is not None else range(pdf_document.page_count)
        )

        matches = []
        for page_number in pages:
            page = pdf_document.load_page(page_number)
            if debug:
                print(f"Page {page_number + 1}:")

            # Extract text blocks with their styles
            for block in page.get_text("dict")["blocks"]:

                if "lines" in block:
                    block_position = block[
                        "bbox"
                    ]  # Position of the block (x0, y0, x1, y1)
                    for line in block["lines"]:
                        for span in line["spans"]:
                            # TODO: optional color?
                            text = span["text"]
                            font_size = span["size"]
                            color = span["color"]
                            if re.match(pattern, text):
                                matches.append(text)
        return matches
    except:
        raise


def adjust(n):
    """
    Only takes first _word_.
    We cannot handle '(' or ')' so we remove.
    We need a prefix. If it's not '-' we add '_'.
    """
    n = n.split(" ")[0].replace("(", "").replace(")", "")
    if n[0] != "-":
        n = "_" + n
    return n


def get_labels(p, input_pdf_file):
    """
    Get all text fields fitting the pattern from page `p` (counting from 1).
    """
    input_pdf = input_pdf_file  # Replace with your input PDF file path
    pattern = "(-.{1,9})|([A-Z]{1,2}[0-9]{1,2})$"
    """
    anything starting from '-'
    OR
    1-2 letters with 1-2 digits
    
    ( -.{1,9} ) 
    | 
    ( [A-Z]{1,2}[0-9]{1,2} )
    $
    """

    groups = find_pattern(input_pdf, pattern, p - 1)
    return list(map(adjust, groups))


def get_type(input_pdf_file, p, label):
    """
    Finds page info using label as a location guide.
    Page `p` counts from 1.
    """
    input_pdf = input_pdf_file  # Replace with your input PDF file path
    # label = "BD.43 Bühne Süd TKF Scheibenkleben"

    page_to_extract = p - 1
    groups = find_label_info(input_pdf, page_number=page_to_extract, label=label)
    return groups


def get_page_elements(input_pdf_file, label, pages):
    """
    Gets all page text fields fitting the patterns with page annotation info.
    """
    pages_metadata = {}
    for p in pages:
        pages_metadata[p] = {
            "page_annotation": "".join(get_type(input_pdf_file, p, label)).replace(
                " ", ""
            ),
            "labels": get_labels(p, input_pdf_file),
        }
    return pages_metadata


def process_page_elements_labels(data):
    """
    Joins pages info into label-centric list.
    """
    labels = {}
    for k in data:
        pa = data[k]["page_annotation"]
        for n in data[k]["labels"]:
            l = pa + n
            labels[l] = {"label": l, "pages": labels.get(l, {}).get("pages", []) + [k]}
    return list(labels.values())


def get_pages_elements(config: TextExtracionConfig):
    filename = config.filename
    prefix = config.prefix
    pages = range(*config.pages)
    label = config.context_label
    data = get_page_elements(filename, label, pages)
    data = process_page_elements_labels(data)

    if prefix:
        for data_point in data:
            data_point["label"] = prefix + data_point["label"]

    return data
