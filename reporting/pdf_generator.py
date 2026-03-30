"""
reporting/pdf_generator.py
==========================
Generates a professional PDF report using ReportLab (Platypus).
Uses native ReportLab charts — no kaleido or Plotly required.
"""

from __future__ import annotations

import os
from datetime import date
from typing import List, Any, Dict

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    PageBreak,
    KeepTogether,
)
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.piecharts import Pie

import config
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_ACCENT     = colors.HexColor("#2980B9")
_DARK       = colors.HexColor("#2C3E50")
_LIGHT_GREY = colors.HexColor("#ECF0F1")
_WHITE      = colors.white
_NEG_RED    = colors.HexColor("#E74C3C")
_POS_GREEN  = colors.HexColor("#27AE60")


class PDFReportGenerator:
    """
    Builds a polished PDF report for the given review analysis data.

    Parameters
    ----------
    reviews_df : pd.DataFrame
        Analysed reviews (must have sentiment columns).
    trend_df : pd.DataFrame
        Output of TrendAnalyser.compute().
    issues_df : pd.DataFrame
        Output of IssuePrioritizer.prioritize().
    alerts : List[dict]
        Alert dicts from TrendAnalyser.
    date_from : date
        Start of the reporting period.
    date_to : date
        End of the reporting period.
    output_path : str
        Where to save the PDF.
    company_name : str
        Shown on the cover page.
    """

    def __init__(
        self,
        reviews_df: pd.DataFrame,
        trend_df: pd.DataFrame,
        issues_df: pd.DataFrame,
        alerts: List[dict],
        date_from: date,
        date_to: date,
        output_path: str,
        company_name: str = config.PDF_COMPANY_NAME,
    ) -> None:
        self.reviews_df  = reviews_df
        self.trend_df    = trend_df
        self.issues_df   = issues_df
        self.alerts      = alerts
        self.date_from   = date_from
        self.date_to     = date_to
        self.output_path = output_path
        self.company_name = company_name
        self._styles     = self._build_styles()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(self) -> str:
        """
        Build and save the PDF report.

        Returns
        -------
        str
            Absolute path to the saved PDF file.
        """
        os.makedirs(
            os.path.dirname(os.path.abspath(self.output_path)),
            exist_ok=True,
        )

        doc = BaseDocTemplate(
            self.output_path,
            pagesize=A4,
            leftMargin=2 * cm,
            rightMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
        )

        frame = Frame(
            doc.leftMargin,
            doc.bottomMargin,
            doc.width,
            doc.height,
            id="main",
        )
        doc.addPageTemplates(
            [
                PageTemplate(
                    id="main",
                    frames=frame,
                    onPage=self._add_header_footer,
                )
            ]
        )

        story = self._build_story()
        doc.build(story)
        logger.info("PDF report saved to '%s'.", self.output_path)
        return self.output_path

    # ------------------------------------------------------------------
    # Story builder
    # ------------------------------------------------------------------

    def _build_story(self) -> list:
        """Assemble all flowable elements into the report."""
        story: list = []
        story += self._cover_page()
        story.append(PageBreak())
        story += self._executive_summary()
        story.append(Spacer(1, 0.5 * cm))
        story += self._trend_chart_section()
        story.append(Spacer(1, 0.5 * cm))
        story += self._distribution_chart_section()
        story.append(Spacer(1, 0.5 * cm))
        story += self._top_issues_section()
        story.append(Spacer(1, 0.5 * cm))
        story += self._alerts_section()
        story.append(Spacer(1, 0.5 * cm))
        story += self._recommendations_section()
        return story

    # ------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------

    def _cover_page(self) -> list:
        s = self._styles
        return [
            Spacer(1, 3 * cm),
            Paragraph(self.company_name, s["CoverSubtitle"]),
            Spacer(1, 0.5 * cm),
            Paragraph("Feedback Intelligence Report", s["CoverTitle"]),
            Spacer(1, 0.5 * cm),
            HRFlowable(width="100%", thickness=3, color=_ACCENT),
            Spacer(1, 0.5 * cm),
            Paragraph(
                f"Reporting Period: "
                f"{self.date_from.strftime('%B %d, %Y')} — "
                f"{self.date_to.strftime('%B %d, %Y')}",
                s["CoverMeta"],
            ),
            Paragraph(
                f"Generated: {date.today().strftime('%B %d, %Y')}",
                s["CoverMeta"],
            ),
            Spacer(1, 1 * cm),
            Paragraph(
                f"Total Reviews Analysed: <b>{len(self.reviews_df):,}</b>",
                s["CoverMeta"],
            ),
        ]

    def _executive_summary(self) -> list:
        s = self._styles
        df = self.reviews_df
        total = len(df)
        positive = int((df["sentiment_label"] == "POSITIVE").sum()) \
            if "sentiment_label" in df.columns else 0
        negative = total - positive
        avg_rating = float(df["rating"].mean()) \
            if "rating" in df.columns else 0.0
        avg_sentiment = float(df["sentiment_score"].mean()) \
            if "sentiment_score" in df.columns else 0.0

        kpi_data = [
            ["Metric", "Value"],
            ["Total Reviews", f"{total:,}"],
            ["Positive Reviews",
             f"{positive:,} ({100 * positive / total:.1f}%)" if total else "0"],
            ["Negative Reviews",
             f"{negative:,} ({100 * negative / total:.1f}%)" if total else "0"],
            ["Average Star Rating", f"{avg_rating:.2f} / 5.0"],
            ["Average Sentiment Score", f"{avg_sentiment:+.3f}"],
            ["Sentiment Alerts", str(len(self.alerts))],
        ]

        table = Table(kpi_data, colWidths=[10 * cm, 7 * cm])
        table.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1,  0), _ACCENT),
            ("TEXTCOLOR",     (0, 0), (-1,  0), _WHITE),
            ("FONTNAME",      (0, 0), (-1,  0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1,  0), 11),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [_WHITE, _LIGHT_GREY]),
            ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",      (0, 1), (-1, -1), 10),
            ("ALIGN",         (1, 0), (-1, -1), "CENTER"),
            ("GRID",          (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING",       (0, 0), (-1, -1), 6),
        ]))

        return [
            Paragraph("Executive Summary", s["SectionHeader"]),
            HRFlowable(width="100%", thickness=1, color=_ACCENT),
            Spacer(1, 0.3 * cm),
            KeepTogether([table]),
        ]

    def _trend_chart_section(self) -> list:
        """Render sentiment trend using native ReportLab HorizontalLineChart."""
        s = self._styles
        elements: list = [
            Paragraph("Sentiment Trend", s["SectionHeader"]),
            HRFlowable(width="100%", thickness=1, color=_ACCENT),
            Spacer(1, 0.3 * cm),
        ]

        if self.trend_df.empty:
            elements.append(Paragraph("No trend data available.", s["Body"]))
            return elements

        try:
            data = (
                self.trend_df["rolling_avg_sentiment"]
                .fillna(0)
                .round(3)
                .tolist()
            )

            drawing = Drawing(450, 180)
            chart = HorizontalLineChart()
            chart.x = 50
            chart.y = 20
            chart.width = 370
            chart.height = 140
            chart.data = [data]
            chart.lines[0].strokeColor = _ACCENT
            chart.lines[0].strokeWidth = 2
            chart.valueAxis.valueMin = max(-1.0, min(data) - 0.1)
            chart.valueAxis.valueMax = min(1.0,  max(data) + 0.1)
            chart.valueAxis.valueStep = 0.2
            chart.valueAxis.labels.fontSize = 7

            dates = self.trend_df["date"].tolist()
            step = max(1, len(dates) // 6)
            chart.categoryAxis.categoryNames = [
                str(dates[i])[:10] if i % step == 0 else ""
                for i in range(len(dates))
            ]
            chart.categoryAxis.labels.fontSize = 6
            chart.categoryAxis.labels.angle = 30
            chart.categoryAxis.labels.boxAnchor = "ne"

            drawing.add(chart)
            elements.append(drawing)

        except Exception as exc:
            logger.warning("Trend chart rendering failed: %s", exc)
            elements.append(
                Paragraph("Trend chart could not be rendered.", s["Body"])
            )

        return elements

    def _distribution_chart_section(self) -> list:
        """Render sentiment pie chart using native ReportLab Pie."""
        s = self._styles
        elements: list = [
            Paragraph("Sentiment Distribution", s["SectionHeader"]),
            HRFlowable(width="100%", thickness=1, color=_ACCENT),
            Spacer(1, 0.3 * cm),
        ]

        if "sentiment_label" not in self.reviews_df.columns:
            elements.append(Paragraph("No sentiment data.", s["Body"]))
            return elements

        try:
            dist   = self.reviews_df["sentiment_label"].value_counts()
            labels = dist.index.tolist()
            values = dist.values.tolist()

            drawing = Drawing(300, 160)
            pie = Pie()
            pie.x = 70
            pie.y = 10
            pie.width  = 130
            pie.height = 130
            pie.data   = values
            pie.labels = [f"{l} ({v})" for l, v in zip(labels, values)]
            pie.slices.strokeWidth = 0.5

            slice_colors = [_POS_GREEN, _NEG_RED, _ACCENT]
            for i, col in enumerate(slice_colors[: len(labels)]):
                pie.slices[i].fillColor = col

            drawing.add(pie)
            elements.append(drawing)

        except Exception as exc:
            logger.warning("Distribution chart rendering failed: %s", exc)
            elements.append(
                Paragraph("Distribution chart could not be rendered.", s["Body"])
            )

        return elements

    def _top_issues_section(self) -> list:
        s = self._styles
        elements: list = [
            Paragraph("Top Issues", s["SectionHeader"]),
            HRFlowable(width="100%", thickness=1, color=_ACCENT),
            Spacer(1, 0.3 * cm),
        ]

        if self.issues_df.empty:
            elements.append(Paragraph("No issues identified.", s["Body"]))
            return elements

        display = self.issues_df.head(10)
        data = [["Rank", "Keyword", "Frequency",
                  "Avg Neg Strength", "Priority Score"]]
        for _, row in display.iterrows():
            data.append([
                str(int(row["rank"])),
                str(row["keyword"]).title(),
                str(int(row["frequency"])),
                f"{row['avg_neg_strength']:.3f}",
                f"{row['priority_score']:.3f}",
            ])

        col_widths = [1.5 * cm, 5 * cm, 3 * cm, 4 * cm, 4 * cm]
        table = Table(data, colWidths=col_widths)
        table.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1,  0), _DARK),
            ("TEXTCOLOR",     (0, 0), (-1,  0), _WHITE),
            ("FONTNAME",      (0, 0), (-1,  0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1,  0), 9),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [_WHITE, _LIGHT_GREY]),
            ("FONTSIZE",      (0, 1), (-1, -1), 9),
            ("ALIGN",         (0, 0), ( 0, -1), "CENTER"),
            ("ALIGN",         (2, 0), (-1, -1), "CENTER"),
            ("GRID",          (0, 0), (-1, -1), 0.3, colors.grey),
            ("PADDING",       (0, 0), (-1, -1), 5),
        ]))

        elements.append(KeepTogether([table]))
        return elements

    def _alerts_section(self) -> list:
        s = self._styles
        elements: list = [
            Paragraph("Sentiment Alerts", s["SectionHeader"]),
            HRFlowable(width="100%", thickness=1, color=_ACCENT),
            Spacer(1, 0.3 * cm),
        ]

        if not self.alerts:
            elements.append(
                Paragraph(
                    "No sentiment alerts during this period.", s["Body"]
                )
            )
        else:
            for alert in self.alerts:
                elements.append(
                    Paragraph(f"• {alert['message']}", s["AlertText"])
                )
        return elements

    def _recommendations_section(self) -> list:
        s = self._styles
        elements: list = [
            Paragraph("Recommendations", s["SectionHeader"]),
            HRFlowable(width="100%", thickness=1, color=_ACCENT),
            Spacer(1, 0.3 * cm),
        ]
        for i, rec in enumerate(self._generate_recommendations(), 1):
            elements.append(Paragraph(f"{i}. {rec}", s["Body"]))
            elements.append(Spacer(1, 0.2 * cm))
        return elements

    # ------------------------------------------------------------------
    # Recommendation engine
    # ------------------------------------------------------------------

    def _generate_recommendations(self) -> List[str]:
        """Generate data-driven recommendations from analysis results."""
        recs: List[str] = []
        df = self.reviews_df

        if "sentiment_score" in df.columns:
            avg = float(df["sentiment_score"].mean())
            if avg < -0.2:
                recs.append(
                    "Overall sentiment is significantly negative. Prioritise a "
                    "rapid response plan addressing the top-ranked complaint keywords."
                )
            elif avg > 0.3:
                recs.append(
                    "Sentiment is broadly positive. Focus on maintaining quality "
                    "and amplifying positive experiences through testimonials."
                )

        if self.alerts:
            recs.append(
                f"There were {len(self.alerts)} sentiment drop alert(s) this period. "
                "Investigate related releases, support tickets, or incidents."
            )

        if not self.issues_df.empty:
            top_kw = self.issues_df.iloc[0]["keyword"]
            recs.append(
                f"The highest-priority issue keyword is '{str(top_kw).title()}'. "
                "Assign an owner and create a sprint ticket to address it."
            )

        if "source" in df.columns and df["source"].nunique() > 1:
            worst = (
                df.groupby("source")["sentiment_score"].mean().idxmin()
            )
            recs.append(
                f"Reviews from '{worst}' have the lowest average sentiment. "
                "Consider a platform-specific improvement initiative."
            )

        if not recs:
            recs.append(
                "No critical issues detected. Continue monitoring weekly."
            )

        return recs

    # ------------------------------------------------------------------
    # Header / footer
    # ------------------------------------------------------------------

    @staticmethod
    def _add_header_footer(canvas: Any, doc: Any) -> None:
        """Draw page header and footer on every page."""
        canvas.saveState()
        page_width, page_height = A4

        canvas.setStrokeColor(_ACCENT)
        canvas.setLineWidth(1.5)
        canvas.line(
            2 * cm, page_height - 1.5 * cm,
            page_width - 2 * cm, page_height - 1.5 * cm,
        )
        canvas.setFont("Helvetica-Bold", 8)
        canvas.setFillColor(_ACCENT)
        canvas.drawString(
            2 * cm, page_height - 1.3 * cm,
            "Feedback Intelligence System",
        )

        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.grey)
        canvas.drawRightString(
            page_width - 2 * cm, 1.2 * cm,
            f"Page {doc.page} | CONFIDENTIAL",
        )
        canvas.restoreState()

    # ------------------------------------------------------------------
    # Styles
    # ------------------------------------------------------------------

    @staticmethod
    def _build_styles() -> dict:
        styles = {
            "CoverTitle": ParagraphStyle(
                "CoverTitle",
                fontSize=28,
                fontName="Helvetica-Bold",
                textColor=_DARK,
                alignment=TA_CENTER,
                spaceAfter=12,
            ),
            "CoverSubtitle": ParagraphStyle(
                "CoverSubtitle",
                fontSize=16,
                fontName="Helvetica",
                textColor=_ACCENT,
                alignment=TA_CENTER,
                spaceAfter=6,
            ),
            "CoverMeta": ParagraphStyle(
                "CoverMeta",
                fontSize=11,
                fontName="Helvetica",
                textColor=colors.grey,
                alignment=TA_CENTER,
                spaceAfter=4,
            ),
            "SectionHeader": ParagraphStyle(
                "SectionHeader",
                fontSize=14,
                fontName="Helvetica-Bold",
                textColor=_DARK,
                spaceBefore=12,
                spaceAfter=4,
            ),
            "Body": ParagraphStyle(
                "Body",
                fontSize=10,
                fontName="Helvetica",
                textColor=_DARK,
                alignment=TA_JUSTIFY,
                spaceAfter=4,
            ),
            "AlertText": ParagraphStyle(
                "AlertText",
                fontSize=10,
                fontName="Helvetica",
                textColor=_NEG_RED,
                spaceAfter=4,
            ),
        }
        return styles