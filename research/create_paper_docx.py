from pathlib import Path
import re

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "research" / "paper.md"
OUTPUT = ROOT / "research" / "paper.docx"


def set_cell_shading(cell, fill):
    properties = cell._tc.get_or_add_tcPr()
    shading = properties.find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        properties.append(shading)
    shading.set(qn("w:fill"), fill)


def set_repeat_table_header(row):
    properties = row._tr.get_or_add_trPr()
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    properties.append(repeat)


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    field_begin = OxmlElement("w:fldChar")
    field_begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = " PAGE "
    field_end = OxmlElement("w:fldChar")
    field_end.set(qn("w:fldCharType"), "end")
    run._r.append(field_begin)
    run._r.append(instruction)
    run._r.append(field_end)


def clean_inline(text):
    text = re.sub(r"!\[[^]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", text)
    text = text.replace("**", "").replace("__", "")
    text = text.replace("`", "")
    text = text.replace("\\", "")
    return text.strip()


def add_body_paragraph(document, text, style="Normal"):
    paragraph = document.add_paragraph(style=style)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    paragraph.paragraph_format.space_after = Pt(5)
    paragraph.paragraph_format.line_spacing = 1.08
    paragraph.add_run(clean_inline(text))
    return paragraph


def add_caption(document, text):
    paragraph = document.add_paragraph(style="Caption")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(2)
    paragraph.paragraph_format.space_after = Pt(8)
    run = paragraph.add_run(clean_inline(text))
    run.bold = True
    return paragraph


def add_markdown_table(document, rows):
    if len(rows) < 2:
        return
    data = [row[1:-1] if row.startswith("|") else row.split("|") for row in rows]
    data = [[clean_inline(cell.strip()) for cell in row] for row in data]
    if len(data) > 1 and all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in data[1]):
        data.pop(1)
    columns = max(len(row) for row in data)
    table = document.add_table(rows=len(data), cols=columns)
    table.style = "Table Grid"
    table.autofit = True
    for row_index, values in enumerate(data):
        row = table.rows[row_index]
        if row_index == 0:
            set_repeat_table_header(row)
        for column_index in range(columns):
            cell = row.cells[column_index]
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            value = values[column_index] if column_index < len(values) else ""
            cell.text = value
            set_cell_shading(cell, "D9EAF7" if row_index == 0 else "FFFFFF")
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_after = Pt(0)
                for run in paragraph.runs:
                    run.font.name = "Times New Roman"
                    run.font.size = Pt(8)
                    run.bold = row_index == 0
    document.add_paragraph().paragraph_format.space_after = Pt(1)


def image_for_markdown(target):
    name = Path(target.split("#")[0]).name
    mapping = {
        "plantvillage_vs_external_accuracy.png": ROOT / "results" / "final_figures" / "plantvillage_vs_external_accuracy.png",
        "plantvillage_vs_external_macro_f1.png": ROOT / "results" / "final_figures" / "plantvillage_vs_external_macro_f1.png",
        "domain_gap_accuracy.png": ROOT / "results" / "final_figures" / "domain_gap_accuracy.png",
        "parameter_count_vs_latency.png": ROOT / "results" / "final_figures" / "parameter_count_comparison.png",
        "selected_gradcam_examples.png": ROOT / "results" / "final_figures" / "explainability_examples.png",
        "plantdoc_confidence_correct_vs_incorrect.png": ROOT / "results" / "final_figures" / "plantdoc_confidence_correct_vs_incorrect.png",
        "plantseg_v7_confidence_correct_vs_incorrect.png": ROOT / "results" / "final_figures" / "plantseg_v7_confidence_correct_vs_incorrect.png",
    }
    return mapping.get(name)


def create_schematic_figures():
    output_dir = ROOT / "results" / "final_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = output_dir / "leafguard_pipeline.png"
    architecture = output_dir / "hybrid_architecture.png"
    if not pipeline.exists():
        fig, axis = plt.subplots(figsize=(8, 6), dpi=220)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.axis("off")
        labels = [
            "PlantVillage-derived training split",
            "Four frozen checkpoints",
            "PlantVillage test evaluation",
            "PlantDoc and PlantSeg mapped evaluation",
            "Failure, confidence, Grad-CAM++, and cost analysis",
        ]
        y_values = [0.88, 0.70, 0.52, 0.34, 0.16]
        for index, (label, y_value) in enumerate(zip(labels, y_values)):
            box = FancyBboxPatch((0.12, y_value - 0.05), 0.76, 0.10, boxstyle="round,pad=0.02", facecolor="#D9EAF7", edgecolor="#2F5597", linewidth=1.5)
            axis.add_patch(box)
            axis.text(0.50, y_value, label, ha="center", va="center", fontsize=10)
            if index < len(labels) - 1:
                axis.annotate("", xy=(0.50, y_values[index + 1] + 0.055), xytext=(0.50, y_value - 0.055), arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#2F5597"})
        fig.savefig(pipeline, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    if not architecture.exists():
        fig, axis = plt.subplots(figsize=(8, 4.2), dpi=220)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.axis("off")
        boxes = [(0.06, 0.65, "MobileNetV3\n576 features"), (0.06, 0.18, "DeiT-Tiny\n192 features"), (0.43, 0.42, "Feature fusion\n768 dimensions"), (0.73, 0.42, "Classifier\n768 -> 512 -> 256 -> 10")]
        for x_value, y_value, label in boxes:
            box = FancyBboxPatch((x_value, y_value), 0.21, 0.18, boxstyle="round,pad=0.02", facecolor="#E2F0D9", edgecolor="#548235", linewidth=1.5)
            axis.add_patch(box)
            axis.text(x_value + 0.105, y_value + 0.09, label, ha="center", va="center", fontsize=10)
        axis.annotate("", xy=(0.43, 0.51), xytext=(0.27, 0.74), arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#548235"})
        axis.annotate("", xy=(0.43, 0.51), xytext=(0.27, 0.27), arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#548235"})
        axis.annotate("", xy=(0.73, 0.51), xytext=(0.64, 0.51), arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#548235"})
        fig.savefig(architecture, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return pipeline, architecture


def add_image(document, target, caption):
    path = image_for_markdown(target)
    if path is None or not path.exists():
        return
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run().add_picture(str(path), width=Inches(6.1))
    add_caption(document, caption)


def setup_document():
    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(0.7)
    section.bottom_margin = Inches(0.7)
    section.left_margin = Inches(0.8)
    section.right_margin = Inches(0.8)

    styles = document.styles
    normal = styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10.5)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    for name, size, color in (("Heading 1", 14, "1F4E79"), ("Heading 2", 11, "2F5597")):
        style = styles[name]
        style.font.name = "Times New Roman"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
    caption = styles["Caption"]
    caption.font.name = "Times New Roman"
    caption.font.size = Pt(9)
    caption.font.italic = True
    footer = section.footer.paragraphs[0]
    footer.text = "LeafGuard Research Manuscript | "
    footer.runs[0].font.name = "Times New Roman"
    footer.runs[0].font.size = Pt(8)
    add_page_number(footer)
    return document


def convert():
    pipeline_figure, architecture_figure = create_schematic_figures()
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    document = setup_document()
    index = 0
    title_done = False
    in_code = False
    code_lines = []
    paragraph_lines = []

    def flush_paragraph():
        nonlocal paragraph_lines
        if paragraph_lines:
            text = " ".join(line.strip() for line in paragraph_lines).strip()
            if text:
                add_body_paragraph(document, text)
            paragraph_lines = []

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if stripped == "```":
            flush_paragraph()
            if not in_code:
                in_code = True
                code_lines = []
            else:
                code_text = "\n".join(code_lines)
                if "PlantVillage Training Split" in code_text:
                    paragraph = document.add_paragraph()
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    paragraph.add_run().add_picture(str(pipeline_figure), width=Inches(6.0))
                    add_caption(document, "Fig. 1. Overall LeafGuard cross-domain research and evaluation pipeline.")
                elif "MobileNetV3 Branch" in code_text:
                    paragraph = document.add_paragraph()
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    paragraph.add_run().add_picture(str(architecture_figure), width=Inches(6.0))
                    add_caption(document, "Fig. 2. Architectural overview of the dual-branch MobileNetV3-DeiT-Tiny feature-fusion hybrid.")
                else:
                    paragraph = document.add_paragraph()
                    paragraph.style = "No Spacing"
                    paragraph.paragraph_format.left_indent = Inches(0.25)
                    run = paragraph.add_run(code_text)
                    run.font.name = "Courier New"
                    run.font.size = Pt(8)
                in_code = False
            index += 1
            continue
        if in_code:
            code_lines.append(line)
            index += 1
            continue
        if not stripped:
            flush_paragraph()
            index += 1
            continue
        if not title_done and line.startswith("# "):
            flush_paragraph()
            paragraph = document.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.space_after = Pt(10)
            run = paragraph.add_run(line[2:].strip())
            run.bold = True
            run.font.name = "Times New Roman"
            run.font.size = Pt(17)
            title_done = True
            index += 1
            continue
        if line.startswith("## "):
            flush_paragraph()
            heading = document.add_heading(clean_inline(line[3:]), level=1)
            heading.paragraph_format.page_break_before = line.startswith("## I.") or line.startswith("## IV.") or line.startswith("## V.")
            index += 1
            continue
        if line.startswith("### "):
            flush_paragraph()
            document.add_heading(clean_inline(line[4:]), level=2)
            index += 1
            continue
        if line.startswith("TABLE ") or line.startswith("**TABLE "):
            flush_paragraph()
            add_caption(document, line.strip("*") + (" " + lines[index + 1].strip("*") if index + 1 < len(lines) and not lines[index + 1].startswith("|") else ""))
            index += 2 if index + 1 < len(lines) and not lines[index + 1].startswith("|") else 1
            continue
        if line.startswith("|") and index + 1 < len(lines) and lines[index + 1].startswith("|"):
            flush_paragraph()
            table_rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_rows.append(lines[index].strip())
                index += 1
            add_markdown_table(document, table_rows)
            continue
        image_match = re.match(r"!\[[^]]*\]\(([^)]+)\)", stripped)
        if image_match:
            flush_paragraph()
            caption = lines[index + 1].strip() if index + 1 < len(lines) else ""
            add_image(document, image_match.group(1), caption)
            index += 2 if caption.startswith("**Fig.") else 1
            continue
        if stripped.startswith(">"):
            flush_paragraph()
            note = document.add_paragraph()
            note.paragraph_format.left_indent = Inches(0.25)
            run = note.add_run(clean_inline(stripped.lstrip("> ")))
            run.italic = True
            run.font.name = "Times New Roman"
            run.font.size = Pt(9)
            index += 1
            continue
        if re.match(r"^(\d+\.|[-*])\s+", stripped):
            flush_paragraph()
            text = re.sub(r"^(\d+\.|[-*])\s+", "", stripped)
            paragraph = document.add_paragraph(style="List Number" if stripped[0].isdigit() else "List Bullet")
            paragraph.add_run(clean_inline(text))
            index += 1
            continue
        if stripped == "---":
            flush_paragraph()
            document.add_paragraph().add_run().add_break(WD_BREAK.LINE)
            index += 1
            continue
        paragraph_lines.append(line)
        index += 1
    flush_paragraph()
    document.core_properties.title = "Cross-Dataset Evaluation of Plant-Disease Classifiers Under Domain Shift"
    document.core_properties.subject = "LeafGuard research manuscript"
    document.core_properties.author = "LeafGuard Research Team"
    document.save(OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    convert()