import fitz
from pydantic import BaseModel


class Span(BaseModel):
    size: float
    flags: int
    font: str
    color: int
    ascender: float
    descender: float
    text: str
    origin: tuple
    bbox: tuple


class Line(BaseModel):
    spans: list[Span]
    wmode: int
    dir: tuple
    bbox: tuple


class Block(BaseModel):
    number: int
    type: int
    bbox: tuple
    lines: list[Line]


class Page(fitz.Page): ...


class PageComponents(BaseModel):
    text_blocks: list
    text_positions: list

    def get_blocks(self):
        return zip(self.text_blocks, self.text_positions)
