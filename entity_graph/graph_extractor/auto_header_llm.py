"""
Automatic Header Extraction Module for ContextClue Graph Builder (LLM-enabled)

Adds an optional LLM refinement stage on top of the improved module:
- Pluggable `LLMClient` protocol with a thin `OpenAIClient` wrapper
- `LLMHeaderAdvisor` that re-ranks, merges, and normalizes header candidates
- Toggle via `use_llm=True` and pass `llm_advisor=LLMHeaderAdvisor(...)`

This file is standalone and includes the improved extractor code plus LLM hooks.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Optional dependencies with feature flags
# -----------------------------------------------------------------------------
try:
    import fitz  # PyMuPDF
    HAVE_PYMUPDF = True
except Exception:
    fitz = None
    HAVE_PYMUPDF = False

try:
    import cv2  # type: ignore
    HAVE_OPENCV = True
except Exception:
    cv2 = None
    HAVE_OPENCV = False

try:
    import pytesseract  # type: ignore
    HAVE_TESSERACT = True
except Exception:
    pytesseract = None
    HAVE_TESSERACT = False

try:
    import spacy  # type: ignore
    HAVE_SPACY = True
except Exception:
    spacy = None
    HAVE_SPACY = False

try:
    from transformers import AutoModel, AutoTokenizer  # type: ignore
    import torch  # type: ignore
    HAVE_TRANSFORMERS = True
except Exception:
    AutoModel = AutoTokenizer = torch = None
    HAVE_TRANSFORMERS = False

LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Enums & Data Models
# -----------------------------------------------------------------------------
class HeaderExtractionMethod(Enum):
    LAYOUT_ANALYSIS = "layout_analysis"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    PATTERN_RECOGNITION = "pattern_recognition"
    HYBRID = "hybrid"


@dataclass(order=True)
class HeaderCandidate:
    sort_index: float = field(init=False, repr=False)
    text: str
    confidence: float
    position: Tuple[float, float, float, float]  # x1, y1, x2, y2
    method: str
    semantic_score: float = 0.0
    layout_score: float = 0.0
    pattern_score: float = 0.0
    page: Optional[int] = None

    def __post_init__(self):
        self.sort_index = -float(self.confidence)


@dataclass
class TableRegion:
    bbox: Tuple[float, float, float, float]
    headers: List[HeaderCandidate]
    confidence: float
    page_number: int


@dataclass
class HeaderExtractionResult:
    headers: List[str]
    candidates: List[HeaderCandidate]
    table_regions: List[TableRegion]
    method: HeaderExtractionMethod
    confidence_threshold: float
    file_path: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "headers": self.headers,
            "candidates": [
                {
                    "text": c.text,
                    "confidence": c.confidence,
                    "position": tuple(c.position),
                    "method": c.method,
                    "semantic_score": c.semantic_score,
                    "layout_score": c.layout_score,
                    "pattern_score": c.pattern_score,
                    "page": c.page,
                }
                for c in self.candidates
            ],
            "table_regions": [
                {
                    "bbox": tuple(tr.bbox),
                    "headers": [h.text for h in tr.headers],
                    "confidence": tr.confidence,
                    "page_number": tr.page_number,
                }
                for tr in self.table_regions
            ],
            "method": self.method.value,
            "confidence_threshold": self.confidence_threshold,
            "file_path": self.file_path,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


# -----------------------------------------------------------------------------
# Semantic Classifier
# -----------------------------------------------------------------------------
class SemanticHeaderClassifier:
    def __init__(self, model_name: str = "distilbert-base-uncased", use_spacy: bool = True):
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        self._nlp = None
        if use_spacy and HAVE_SPACY:
            try:
                self._nlp = spacy.load("en_core_web_sm")  # type: ignore
            except Exception as e:
                LOGGER.warning("spaCy model not loaded: %s", e)
        self._re_title = re.compile(r"^[A-Z][a-z\s]+$")
        self._re_allcaps_underscores = re.compile(r"^[A-Z0-9_]+$")

    def _ensure_transformers(self):
        if not HAVE_TRANSFORMERS:
            raise RuntimeError("transformers/torch not installed")
        if self._tokenizer is None or self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)  # type: ignore
            self._model = AutoModel.from_pretrained(self.model_name)  # type: ignore

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip()

    def classify_header_semantic(self, text: str) -> float:
        t = self._normalize(text)
        if len(t) < 2:
            return 0.0
        tl = t.lower()
        score = 0.0
        if any(kw in tl for kws in [
            ["id", "identifier", "code", "number", "ref", "reference"],
            ["name", "title", "description", "label", "type", "category"],
            ["quantity", "amount", "count", "volume", "weight", "size"],
            ["attribute", "property", "feature", "characteristic"],
            ["date", "time", "year", "month", "day", "created", "modified"],
            ["price", "cost", "value", "currency", "total"],
            ["specification", "model", "version", "capacity", "power", "performance"],
            ["status", "state", "condition", "active", "enabled", "available"],
            ["country", "city", "address", "location", "region"],
        ] for kw in kws):
            score += 0.3
        score += 0.2 if 3 <= len(t) <= 50 else (-0.1 if len(t) > 50 else 0.0)
        if t.istitle() or t.isupper():
            score += 0.15
        if self._re_title.match(t):
            score += 0.1
        if self._re_allcaps_underscores.match(t):
            score += 0.1
        if "(" in t and ")" in t:
            score += 0.05
        if tl.startswith(("the ", "a ", "an ")):
            score -= 0.1
        if len(t.split()) > 8:
            score -= 0.15
        return float(max(0.0, min(1.0, score)))

    def embedding(self, text: str) -> Optional[np.ndarray]:
        try:
            self._ensure_transformers()
        except Exception as e:
            LOGGER.debug("Embeddings unavailable: %s", e)
            return None
        t = self._normalize(text)
        try:  # type: ignore
            inputs = self._tokenizer(t, return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                outputs = self._model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            return emb.flatten()
        except Exception as e:
            LOGGER.warning("Embedding failed for '%s': %s", t, e)
            return None


# -----------------------------------------------------------------------------
# Layout Analyzer
# -----------------------------------------------------------------------------
class LayoutAnalyzer:
    def __init__(self, y_threshold: float = 5.0):
        self.y_threshold = y_threshold

    @staticmethod
    def _text_regions_pdf(pdf_path: str) -> List[Dict[str, Any]]:
        if not HAVE_PYMUPDF:
            LOGGER.warning("PyMuPDF not installed; PDF layout analysis disabled")
            return []
        regions: List[Dict[str, Any]] = []
        try:
            doc = fitz.open(pdf_path)  # type: ignore
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text_dict = page.get_text("dict")
                for block in text_dict.get("blocks", []):
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = (span.get("text") or "").strip()
                            if not text:
                                continue
                            bbox = tuple(map(float, span.get("bbox", (0, 0, 0, 0))))
                            regions.append({
                                "text": text,
                                "bbox": bbox,
                                "page": page_num,
                                "font_size": float(span.get("size", 0)),
                                "font_flags": int(span.get("flags", 0)),
                                "font": span.get("font", ""),
                            })
            doc.close()
        except Exception as e:
            LOGGER.error("PDF text extraction failed: %s", e)
        return regions

    def detect_table_regions(self, text_regions: List[Dict[str, Any]]) -> List[TableRegion]:
        by_page: Dict[int, List[Dict[str, Any]]] = {}
        for r in text_regions:
            by_page.setdefault(int(r["page"]), []).append(r)
        regions: List[TableRegion] = []
        for page, regs in by_page.items():
            regs.sort(key=lambda x: float(x["bbox"][1]))
            y_positions = [float(r["bbox"][1]) for r in regs]
            clusters = self._cluster(y_positions, self.y_threshold)
            for cluster_y in clusters:
                row_regs = [r for r in regs if abs(float(r["bbox"][1]) - cluster_y) <= self.y_threshold]
                if not self._looks_like_header_row(row_regs):
                    continue
                xs = [r["bbox"][0] for r in row_regs] + [r["bbox"][2] for r in row_regs]
                ys = [r["bbox"][1] for r in row_regs] + [r["bbox"][3] for r in row_regs]
                bbox = (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))
                headers: List[HeaderCandidate] = [
                    HeaderCandidate(
                        text=r["text"],
                        confidence=0.7,
                        position=tuple(map(float, r["bbox"])),
                        method=HeaderExtractionMethod.LAYOUT_ANALYSIS.value,
                        layout_score=0.8,
                        page=page,
                    )
                    for r in row_regs
                ]
                regions.append(TableRegion(bbox=bbox, headers=headers, confidence=0.7, page_number=page))
        return regions

    @staticmethod
    def _cluster(values: Sequence[float], threshold: float) -> List[float]:
        if not values:
            return []
        vs = sorted(values)
        clusters: List[List[float]] = [[vs[0]]]
        for v in vs[1:]:
            if v - clusters[-1][-1] <= threshold:
                clusters[-1].append(v)
            else:
                clusters.append([v])
        return [float(np.mean(c)) for c in clusters if len(c) >= 2]

    @staticmethod
    def _looks_like_header_row(regs: List[Dict[str, Any]]) -> bool:
        if len(regs) < 2:
            return False
        indicators = 0
        for r in regs:
            t = (r.get("text") or "").strip()
            if not t:
                continue
            if (len(t) < 50 and (t.istitle() or t.isupper())) or any(
                k in t.lower() for k in ["id", "name", "type", "date", "value", "number"]
            ):
                indicators += 1
        return indicators >= max(2, int(0.6 * len(regs)))


# -----------------------------------------------------------------------------
# Pattern Recognizer
# -----------------------------------------------------------------------------
class PatternRecognizer:
    def __init__(self):
        self.re_title = re.compile(r"^[A-Z][a-zA-Z\s]+$")
        self.re_allcaps_us = re.compile(r"^[A-Z_]+$")
        self.re_name_unit = re.compile(r"^[A-Z][a-z]+\s*\([^)]+\)$")
        self.re_numbered = re.compile(r"^\d+\.\s*[A-Za-z]")
        self.common = {
            "id","identifier","code","number","name","title","description","type","category","status","date","time","created","modified","value","amount","price","cost","quantity","count","size","weight","height","width","length","volume","capacity","model","version","manufacturer","brand","location","address",
        }

    @staticmethod
    def _norm(t: str) -> str:
        return re.sub(r"\s+", " ", t or "").strip()

    def score(self, text: str) -> float:
        t = self._norm(text)
        if not t:
            return 0.0
        score = 0.0
        if self.re_title.match(t) or self.re_allcaps_us.match(t) or self.re_name_unit.match(t) or self.re_numbered.match(t):
            score += 0.3
        if any(w in t.lower() for w in self.common):
            score += 0.25
        if 3 <= len(t) <= 30:
            score += 0.2
        if t.count(" ") <= 3:
            score += 0.1
        if t.endswith(".") and len(t.split()) > 5:
            score -= 0.25
        return float(max(0.0, min(1.0, score)))


# -----------------------------------------------------------------------------
# LLM Interfaces
# -----------------------------------------------------------------------------
@runtime_checkable
class LLMClient(Protocol):
    def complete(self, prompt: str, *, temperature: float = 0.0, max_tokens: int = 512) -> str:  # pragma: no cover
        ...

class OpenAIClient:
    """Thin OpenAI wrapper (optional). Requires `openai` package and OPENAI_API_KEY."""
    def __init__(self, model: str = "gpt-4o-mini"):
        try:
            import openai  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("openai package not installed") from e
        self._openai = openai
        self._model = model

    def complete(self, prompt: str, *, temperature: float = 0.0, max_tokens: int = 512) -> str:  # pragma: no cover
        resp = self._openai.ChatCompletion.create(
            model=self._model,
            messages=[{"role": "system", "content": "You rewrite column headers for tables."}, {"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp["choices"][0]["message"]["content"].strip()


class LLMHeaderAdvisor:
    DEFAULT_PROMPT = (
        "You help with table header extraction.\n"
        "Given candidate headers and optional context, you will: \n"
        "1) Merge duplicates/synonyms, 2) Prefer concise canonical forms, 3) Preserve semantics (units, currencies),\n"
        "4) Return ONLY JSON: [{\"text\": str, \"weight\": float}] with weights in [0,1].\n\n"
        "Context (optional):\n{context}\n\nCandidates (one per line):\n{cands}\n\nReturn ONLY the JSON."
    )

    def __init__(self, client: LLMClient, *, temperature: float = 0.0, max_tokens: int = 512):
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens

    @staticmethod
    def _make_context(table_regions: List[TableRegion]) -> str:
        lines: List[str] = []
        for tr in table_regions[:2]:
            line = ", ".join(h.text for h in tr.headers[:8])
            lines.append(f"page {tr.page_number}: {line}")
        return "\n".join(lines)

    def advise(self, candidates: List[HeaderCandidate], table_regions: List[TableRegion], *, top_k: int = 40) -> List[HeaderCandidate]:
        if not candidates:
            return []
        cands_sorted = sorted(candidates, key=lambda c: c.confidence, reverse=True)[:top_k]
        cands_text = "\n".join(f"- {c.text} (w={c.confidence:.2f})" for c in cands_sorted)
        context = self._make_context(table_regions)
        prompt = self.DEFAULT_PROMPT.format(context=context or "(none)", cands=cands_text)
        try:
            raw = self.client.complete(prompt, temperature=self.temperature, max_tokens=self.max_tokens)
        except Exception as e:  # pragma: no cover
            LOGGER.warning("LLM refinement failed: %s", e)
            return candidates
        try:
            data = json.loads(raw)
            refined: List[HeaderCandidate] = []
            for item in data:
                txt = str(item.get("text", "")).strip()
                if not txt:
                    continue
                w = float(item.get("weight", 0.5))
                refined.append(HeaderCandidate(text=txt, confidence=max(0.0, min(1.0, w)), position=(0,0,0,0), method="llm_refine"))
            return refined or candidates
        except Exception:
            return candidates


# -----------------------------------------------------------------------------
# Automatic Header Extractor (LLM-enabled)
# -----------------------------------------------------------------------------
class AutomaticHeaderExtractor:
    def __init__(
        self,
        method: HeaderExtractionMethod = HeaderExtractionMethod.HYBRID,
        confidence_threshold: float = 0.6,
        method_weights: Tuple[float, float, float] = (0.4, 0.2, 0.4),  # semantic, layout, pattern
        enable_ocr_fallback: bool = False,
        use_llm: bool = False,
        llm_advisor: Optional[LLMHeaderAdvisor] = None,
        llm_top_k: int = 40,
    ):
        self.method = method
        self.confidence_threshold = confidence_threshold
        self.semantic = SemanticHeaderClassifier()
        self.layout = LayoutAnalyzer()
        self.pattern = PatternRecognizer()
        self.method_weights = method_weights
        self.enable_ocr_fallback = enable_ocr_fallback
        self.use_llm = use_llm
        self.llm_advisor = llm_advisor
        self.llm_top_k = llm_top_k

    # ------------------------------ Public API --------------------------------
    def extract_headers(self, file_path: str, max_headers: int = 20) -> List[str]:
        return self._extract(file_path, max_headers).headers

    def extract(self, file_path: str, max_headers: int = 20) -> HeaderExtractionResult:
        return self._extract(file_path, max_headers)

    def get_extraction_config(self, file_path: str) -> Dict[str, Any]:
        res = self._extract(file_path, max_headers=50)
        return {
            "extraction_type": "table_from_header",
            "filename": file_path,
            "header": res.headers,
            "auto_generated": True,
            "extraction_method": self.method.value,
            "confidence_threshold": self.confidence_threshold,
            "extracted_at": pd.Timestamp.now().isoformat(),
        }

    # ------------------------------ Internals ---------------------------------
    def _extract(self, file_path: str, max_headers: int) -> HeaderExtractionResult:
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        if p.suffix.lower() == ".pdf":
            candidates, regions = self._from_pdf(str(p))
        elif p.suffix.lower() in {".csv", ".xlsx", ".xls"}:
            headers = self._from_tabular(str(p), max_headers)
            candidates = [HeaderCandidate(text=h, confidence=1.0, position=(0, 0, 0, 0), method="tabular") for h in headers]
            regions = []
        else:
            raise ValueError(f"Unsupported file type: {p.suffix}")

        unique = self._dedupe(candidates)
        unique.sort()

        # Optional LLM refinement
        if self.use_llm and self.llm_advisor is not None:
            refined = self.llm_advisor.advise(unique, regions, top_k=self.llm_top_k)
            if refined:
                unique = self._dedupe(refined)
                unique.sort()

        headers = [c.text for c in unique[:max_headers]]
        LOGGER.info("Extracted %d headers", len(headers))
        return HeaderExtractionResult(headers=headers, candidates=unique, table_regions=regions, method=self.method, confidence_threshold=self.confidence_threshold, file_path=str(p))

    def _from_pdf(self, pdf_path: str) -> Tuple[List[HeaderCandidate], List[TableRegion]]:
        if not HAVE_PYMUPDF:
            LOGGER.warning("PyMuPDF not available. PDF extraction limited.")
            return [], []
        text_regions = self.layout._text_regions_pdf(pdf_path)
        if not text_regions and self.enable_ocr_fallback:
            LOGGER.info("No text regions found; attempting OCR fallback")
            self._ocr_pdf_pages(pdf_path)
        table_regions = self.layout.detect_table_regions(text_regions)
        candidates: List[HeaderCandidate] = []
        if self.method in {HeaderExtractionMethod.LAYOUT_ANALYSIS, HeaderExtractionMethod.HYBRID}:
            for tr in table_regions:
                candidates.extend(tr.headers)
        for r in text_regions:
            t = r["text"]
            semantic_score = self.semantic.classify_header_semantic(t) if self.method in {HeaderExtractionMethod.SEMANTIC_SEGMENTATION, HeaderExtractionMethod.HYBRID} else 0.0
            pattern_score = self.pattern.score(t) if self.method in {HeaderExtractionMethod.PATTERN_RECOGNITION, HeaderExtractionMethod.HYBRID} else 0.0
            layout_bonus = 0.0
            if self.method in {HeaderExtractionMethod.LAYOUT_ANALYSIS, HeaderExtractionMethod.HYBRID}:
                fs = float(r.get("font_size", 0))
                if fs >= 9:
                    layout_bonus = 0.05
            if self.method == HeaderExtractionMethod.HYBRID:
                w_sem, w_layout, w_pat = self.method_weights
                combined = w_sem * semantic_score + w_layout * layout_bonus + w_pat * pattern_score
            elif self.method == HeaderExtractionMethod.SEMANTIC_SEGMENTATION:
                combined = semantic_score
            elif self.method == HeaderExtractionMethod.PATTERN_RECOGNITION:
                combined = pattern_score
            else:
                combined = layout_bonus
            combined = float(max(0.0, min(1.0, combined)))
            if combined >= self.confidence_threshold:
                candidates.append(HeaderCandidate(text=t, confidence=combined, position=tuple(map(float, r["bbox"])), method=self.method.value, semantic_score=semantic_score, pattern_score=pattern_score, layout_score=layout_bonus, page=int(r.get("page", -1))))
        return candidates, table_regions

    def _from_tabular(self, path: str, max_headers: int) -> List[str]:
        try:
            if path.endswith(".csv"):
                for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
                    try:
                        df = pd.read_csv(path, nrows=5, encoding=encoding)
                        return self._clean_headers(list(df.columns), max_headers)
                    except Exception:
                        continue
            else:
                df = pd.read_excel(path, nrows=5)
                return self._clean_headers(list(df.columns), max_headers)
        except Exception as e:
            LOGGER.error("Tabular header extraction failed: %s", e)
        return []

    @staticmethod
    def _clean_headers(headers: Iterable[Any], max_headers: int) -> List[str]:
        out: List[str] = []
        for h in headers:
            s = str(h).strip()
            if not s or s.lower().startswith("unnamed") or s.isdigit() or len(s) < 2 or len(s) > 100:
                continue
            out.append(s)
            if len(out) >= max_headers:
                break
        return out

    @staticmethod
    def _norm_key(t: str) -> str:
        t = re.sub(r"\s+", " ", t or "").strip().casefold()
        t = re.sub(r"[^a-z0-9 ]+", "", t)
        return t

    def _dedupe(self, cands: List[HeaderCandidate]) -> List[HeaderCandidate]:
        seen: Dict[str, HeaderCandidate] = {}
        for c in cands:
            k = self._norm_key(c.text)
            if not k:
                continue
            if k not in seen or c.confidence > seen[k].confidence:
                seen[k] = c
        return list(seen.values())

    def _ocr_pdf_pages(self, pdf_path: str) -> None:
        if not (HAVE_OPENCV and HAVE_TESSERACT and HAVE_PYMUPDF):
            LOGGER.info("OCR fallback not available (missing opencv/tesseract/pymupdf)")
            return
        try:
            doc = fitz.open(pdf_path)  # type: ignore
            for i in range(len(doc)):
                pix = doc.load_page(i).get_pixmap(dpi=200)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                if pix.n == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # type: ignore
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # type: ignore
                _ = pytesseract.image_to_string(gray)  # placeholder
            doc.close()
        except Exception as e:
            LOGGER.debug("OCR fallback failed: %s", e)


# -----------------------------------------------------------------------------
# Integration with existing EntitiesGraphExtractor
# -----------------------------------------------------------------------------
class EnhancedEntitiesGraphExtractor:
    def __init__(self, auto_header_extractor: Optional[AutomaticHeaderExtractor] = None):
        try:
            from entity_graph.graph_extractor.entities_graph_extractor import EntitiesGraphExtractor
            self.base_extractor = EntitiesGraphExtractor()  # type: ignore
        except Exception:
            self.base_extractor = None
            LOGGER.warning("Base EntitiesGraphExtractor not available")
        self.auto = auto_header_extractor or AutomaticHeaderExtractor()

    def load_table_from_file_auto(
        self,
        filename: str,
        graph_name: str,
        node_label: str,
        extraction_method: HeaderExtractionMethod = HeaderExtractionMethod.HYBRID,
        confidence_threshold: float = 0.6,
        max_headers: int = 20,
    ) -> Dict[str, Any]:
        self.auto.method = extraction_method
        self.auto.confidence_threshold = confidence_threshold
        config = self.auto.get_extraction_config(filename)
        config.update({"max_headers": max_headers, "graph_name": graph_name, "node_label": node_label})
        if self.base_extractor is None:
            return {**config, "extraction_successful": False, "extraction_error": "Base extractor not available"}
        try:
            result = self.base_extractor.load_table_from_file(config, graph_name, node_label, "instances")  # type: ignore
            return {**config, "extraction_successful": True, "extraction_result": result}
        except Exception as e:
            LOGGER.error("Error in base extractor: %s", e)
            return {**config, "extraction_successful": False, "extraction_error": str(e)}


class EntitiesGraphExtractorWithAutoHeaders:
    def __init__(self, auto_header_extractor: Optional[AutomaticHeaderExtractor] = None):
        self.enhanced = EnhancedEntitiesGraphExtractor(auto_header_extractor)

    def load_table_from_file(self, config: Dict[str, Any], graph_name: str, node_label: str, mode: str):
        if self.enhanced.base_extractor is None:
            raise RuntimeError("Base EntitiesGraphExtractor not available")
        return self.enhanced.base_extractor.load_table_from_file(config, graph_name, node_label, mode)  # type: ignore

    def load_table_from_file_with_auto_headers(
        self,
        filename: str,
        graph_name: str,
        node_label: str,
        extraction_method: HeaderExtractionMethod = HeaderExtractionMethod.HYBRID,
        confidence_threshold: float = 0.6,
        max_headers: int = 20,
        fallback_headers: Optional[List[str]] = None,
        use_llm: bool = False,
        llm_advisor: Optional[LLMHeaderAdvisor] = None,
    ) -> Dict[str, Any]:
        self.enhanced.auto.use_llm = use_llm
        self.enhanced.auto.llm_advisor = llm_advisor
        res = self.enhanced.load_table_from_file_auto(
            filename=filename,
            graph_name=graph_name,
            node_label=node_label,
            extraction_method=extraction_method,
            confidence_threshold=confidence_threshold,
            max_headers=max_headers,
        )
        if not res.get("extraction_successful") and fallback_headers:
            res["header"] = fallback_headers
        return res

    def get_auto_detected_headers(self, filename: str, **kwargs: Any) -> List[str]:
        extractor = self.enhanced.auto
        if kwargs:
            extractor.method = kwargs.get("extraction_method", extractor.method)
            extractor.confidence_threshold = float(kwargs.get("confidence_threshold", extractor.confidence_threshold))
            extractor.use_llm = bool(kwargs.get("use_llm", extractor.use_llm))
            extractor.llm_advisor = kwargs.get("llm_advisor", extractor.llm_advisor)
        return extractor.extract_headers(filename)


# -----------------------------------------------------------------------------
# Standalone utilities
# -----------------------------------------------------------------------------

def extract_headers_from_file(
    file_path: str,
    method: HeaderExtractionMethod = HeaderExtractionMethod.HYBRID,
    confidence_threshold: float = 0.6,
    max_headers: int = 20,
    use_llm: bool = False,
    llm_advisor: Optional[LLMHeaderAdvisor] = None,
) -> List[str]:
    extractor = AutomaticHeaderExtractor(method=method, confidence_threshold=confidence_threshold, use_llm=use_llm, llm_advisor=llm_advisor)
    return extractor.extract_headers(file_path, max_headers)


def generate_auto_config(
    file_path: str,
    method: HeaderExtractionMethod = HeaderExtractionMethod.HYBRID,
    confidence_threshold: float = 0.6,
    use_llm: bool = False,
    llm_advisor: Optional[LLMHeaderAdvisor] = None,
) -> Dict[str, Any]:
    extractor = AutomaticHeaderExtractor(method=method, confidence_threshold=confidence_threshold, use_llm=use_llm, llm_advisor=llm_advisor)
    return extractor.get_extraction_config(file_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== LLM-enabled Header Detection ===")
    # Example (requires openai package + key):
    # advisor = LLMHeaderAdvisor(OpenAIClient(model="gpt-4o-mini"))
    # headers = extract_headers_from_file("sample_document.pdf", use_llm=True, llm_advisor=advisor)
    # print(headers)
