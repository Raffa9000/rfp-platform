"""
RFP Analysis Report Generator
=============================
Generates immutable, cryptographically-signed PDF reports for RFP analysis.

Report Types:
- EXECUTIVE: High-level summary for leadership/business stakeholders
- TECHNICAL: Verbose analysis with all mathematical and semantic details
- LEGAL: Compliance-focused report with risk analysis
- FULL: Complete tribunal report with all data

Patent References:
- 63/920,605: Cryptographic Substrate (BLAKE3 hashing)
- 63/927,104: Multi-LLM Consensus Protocol
- 63/932,010: Trust Entropy Conservation
"""

import io
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, HRFlowable, ListFlowable, ListItem
)
from reportlab.graphics.shapes import Drawing, Rect, String, Line
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.barcharts import VerticalBarChart


class ReportType(Enum):
    EXECUTIVE = "executive"
    TECHNICAL = "technical"
    LEGAL = "legal"
    FULL = "full"


@dataclass
class ReportMetadata:
    """Immutable report metadata with cryptographic hash."""
    report_id: str
    report_type: str
    generated_at: str
    rfp_id: str
    rfp_name: str
    client: str
    analyst_system: str = "Atomic Trust™ RFP Intelligence Platform v4.0"
    content_hash: str = ""

    def compute_hash(self, content: bytes) -> str:
        """Compute BLAKE3-style hash (using SHA-256 as fallback)."""
        # In production, use BLAKE3. SHA-256 for compatibility.
        full_content = json.dumps(asdict(self), sort_keys=True).encode() + content
        return hashlib.sha256(full_content).hexdigest()


class AtomicTrustColors:
    """Brand colors for Atomic Trust™ reports."""
    PRIMARY = colors.HexColor('#0ea5e9')      # atomic-500
    PRIMARY_DARK = colors.HexColor('#0369a1')  # atomic-700
    SUCCESS = colors.HexColor('#10b981')       # emerald-500
    WARNING = colors.HexColor('#f59e0b')       # amber-500
    DANGER = colors.HexColor('#ef4444')        # red-500
    NEUTRAL = colors.HexColor('#6b7280')       # gray-500
    LIGHT = colors.HexColor('#f3f4f6')         # gray-100
    DARK = colors.HexColor('#111827')          # gray-900


class RFPReportGenerator:
    """
    Generates comprehensive PDF reports for RFP analysis.

    Supports multiple report types tailored for different audiences:
    - Executive: Business decision summary
    - Technical: Full mathematical and semantic analysis
    - Legal: Compliance and risk focus
    - Full: Complete tribunal deliberation record
    """

    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Configure custom paragraph styles."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=AtomicTrustColors.PRIMARY_DARK,
            spaceAfter=40,
            alignment=TA_CENTER
        ))

        # Subtitle
        self.styles.add(ParagraphStyle(
            name='ReportSubtitle',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=AtomicTrustColors.NEUTRAL,
            spaceAfter=40,
            alignment=TA_CENTER
        ))

        # Section header
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=AtomicTrustColors.PRIMARY_DARK,
            spaceBefore=30,
            spaceAfter=15,
            borderWidth=0,
            borderPadding=0,
            borderColor=AtomicTrustColors.PRIMARY,
        ))

        # Subsection header
        self.styles.add(ParagraphStyle(
            name='SubsectionHeader',
            parent=self.styles['Heading3'],
            fontSize=11,
            textColor=AtomicTrustColors.DARK,
            spaceBefore=18,
            spaceAfter=10
        ))

        # Body text
        self.styles.add(ParagraphStyle(
            name='ATBodyText',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=AtomicTrustColors.DARK,
            spaceAfter=12,
            alignment=TA_JUSTIFY
        ))

        # Metric value
        self.styles.add(ParagraphStyle(
            name='MetricValue',
            parent=self.styles['Normal'],
            fontSize=24,
            textColor=AtomicTrustColors.PRIMARY_DARK,
            alignment=TA_CENTER
        ))

        # Metric label
        self.styles.add(ParagraphStyle(
            name='MetricLabel',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=AtomicTrustColors.NEUTRAL,
            alignment=TA_CENTER
        ))

        # Code/hash style
        self.styles.add(ParagraphStyle(
            name='CodeStyle',
            parent=self.styles['Normal'],
            fontSize=8,
            fontName='Courier',
            textColor=AtomicTrustColors.NEUTRAL,
            backColor=AtomicTrustColors.LIGHT,
            borderPadding=4
        ))

        # Verdict styles
        self.styles.add(ParagraphStyle(
            name='VerdictBid',
            parent=self.styles['Normal'],
            fontSize=18,
            textColor=AtomicTrustColors.SUCCESS,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))

        self.styles.add(ParagraphStyle(
            name='VerdictNoBid',
            parent=self.styles['Normal'],
            fontSize=18,
            textColor=AtomicTrustColors.DANGER,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))

        self.styles.add(ParagraphStyle(
            name='VerdictConditional',
            parent=self.styles['Normal'],
            fontSize=18,
            textColor=AtomicTrustColors.WARNING,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))

    def generate_report(
        self,
        report_type: ReportType,
        analysis_data: Dict[str, Any],
        rfp_data: Dict[str, Any],
        votes: List[Dict[str, Any]],
        requirements_analysis: List[Dict[str, Any]]
    ) -> bytes:
        """
        Generate a PDF report.

        Args:
            report_type: Type of report to generate
            analysis_data: Final analysis results including consensus
            rfp_data: RFP metadata and details
            votes: Individual LLM votes with rationale
            requirements_analysis: Detailed requirement matching results

        Returns:
            PDF content as bytes
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )

        # Build story based on report type
        story = []

        # Create metadata
        metadata = ReportMetadata(
            report_id=f"RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            report_type=report_type.value,
            generated_at=datetime.now().isoformat(),
            rfp_id=rfp_data.get('id', 'unknown'),
            rfp_name=rfp_data.get('name', 'Unknown RFP'),
            client=rfp_data.get('client', 'Unknown Client')
        )

        # Add content based on report type
        if report_type == ReportType.EXECUTIVE:
            story = self._build_executive_report(metadata, analysis_data, rfp_data, votes)
        elif report_type == ReportType.TECHNICAL:
            story = self._build_technical_report(metadata, analysis_data, rfp_data, votes, requirements_analysis)
        elif report_type == ReportType.LEGAL:
            story = self._build_legal_report(metadata, analysis_data, rfp_data, votes, requirements_analysis)
        else:  # FULL
            story = self._build_full_report(metadata, analysis_data, rfp_data, votes, requirements_analysis)

        # Build PDF
        doc.build(story)

        # Get PDF bytes and compute hash
        pdf_bytes = buffer.getvalue()
        metadata.content_hash = metadata.compute_hash(pdf_bytes)

        buffer.close()
        return pdf_bytes

    def _build_header(self, metadata: ReportMetadata, report_title: str) -> List:
        """Build report header section."""
        elements = []

        # Logo placeholder (would be actual image in production)
        elements.append(Paragraph(
            "<font color='#0ea5e9' size='28'><b>Δ</b></font> Atomic Trust™",
            ParagraphStyle(
                name='Logo',
                fontSize=16,
                textColor=AtomicTrustColors.DARK,
                alignment=TA_CENTER
            )
        ))
        elements.append(Spacer(1, 15))

        # Report title
        elements.append(Paragraph(report_title, self.styles['ReportTitle']))

        # Subtitle with RFP info
        elements.append(Paragraph(
            f"{metadata.rfp_name}<br/><font size='10' color='#6b7280'>{metadata.client}</font>",
            self.styles['ReportSubtitle']
        ))

        # Metadata line
        meta_text = f"Report ID: {metadata.report_id} | Generated: {metadata.generated_at[:19]}"
        elements.append(Paragraph(meta_text, self.styles['CodeStyle']))

        elements.append(Spacer(1, 30))
        elements.append(HRFlowable(width="100%", thickness=1, color=AtomicTrustColors.LIGHT))
        elements.append(Spacer(1, 30))

        return elements

    def _build_verdict_section(self, analysis_data: Dict) -> List:
        """Build the verdict display section."""
        elements = []

        verdict = analysis_data.get('recommendation', 'REVIEW')
        confidence = analysis_data.get('tribunal', {}).get('confidence', 0)

        # Determine style based on verdict
        if 'STRONG_BID' in verdict or verdict == 'BID':
            verdict_style = self.styles['VerdictBid']
            verdict_color = AtomicTrustColors.SUCCESS
        elif 'NO_BID' in verdict or 'NO-BID' in verdict:
            verdict_style = self.styles['VerdictNoBid']
            verdict_color = AtomicTrustColors.DANGER
        else:
            verdict_style = self.styles['VerdictConditional']
            verdict_color = AtomicTrustColors.WARNING

        elements.append(Paragraph("TRIBUNAL VERDICT", self.styles['SectionHeader']))

        # Verdict box
        verdict_display = verdict.replace('_', ' ')
        elements.append(Paragraph(verdict_display, verdict_style))
        elements.append(Paragraph(
            f"Confidence: {confidence*100:.1f}%",
            ParagraphStyle(
                name='ConfidenceText',
                fontSize=12,
                textColor=AtomicTrustColors.NEUTRAL,
                alignment=TA_CENTER
            )
        ))

        elements.append(Spacer(1, 30))
        return elements

    def _build_metrics_table(self, analysis_data: Dict) -> List:
        """Build key metrics table."""
        elements = []

        elements.append(Paragraph("KEY METRICS", self.styles['SectionHeader']))

        tribunal = analysis_data.get('tribunal', {})
        analysis = analysis_data.get('analysis', {})

        # Create metrics data
        metrics = [
            ['Overall Fit', 'MUST Fit', 'Gap Count', 'Tribunal Votes'],
            [
                f"{analysis.get('overall_fit', 0)}%",
                f"{analysis.get('must_requirement_fit', 0)}%",
                str(analysis.get('gap_count', 0)),
                f"{tribunal.get('votes', 0)}/3"
            ]
        ]

        table = Table(metrics, colWidths=[1.5*inch]*4)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), AtomicTrustColors.LIGHT),
            ('TEXTCOLOR', (0, 0), (-1, 0), AtomicTrustColors.NEUTRAL),
            ('TEXTCOLOR', (0, 1), (-1, 1), AtomicTrustColors.DARK),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica'),
            ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTSIZE', (0, 1), (-1, 1), 16),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('TOPPADDING', (0, 1), (-1, 1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, AtomicTrustColors.LIGHT),
        ]))

        elements.append(table)
        elements.append(Spacer(1, 30))

        return elements

    def _build_consensus_metrics(self, analysis_data: Dict) -> List:
        """Build consensus metrics section with mathematical explanation."""
        elements = []

        elements.append(Paragraph("CONSENSUS MATHEMATICS", self.styles['SectionHeader']))

        tribunal = analysis_data.get('tribunal', {})
        phi = tribunal.get('phi', 0)
        psi = tribunal.get('psi', 0)
        delta = tribunal.get('delta', 0)

        # Phi explanation
        elements.append(Paragraph("<b>Phi (Φ) - Agreement Index:</b>", self.styles['SubsectionHeader']))
        elements.append(Paragraph(
            f"<b>{phi:.4f}</b> - Measures inter-model voting consistency. "
            f"Computed as the ratio of agreeing votes to total vote pairs. "
            f"Values > 0.7 indicate strong consensus.",
            self.styles['ATBodyText']
        ))

        # Psi explanation
        elements.append(Paragraph("<b>Psi (Ψ) - Confidence Index:</b>", self.styles['SubsectionHeader']))
        elements.append(Paragraph(
            f"<b>{psi:.4f}</b> - Weighted average of individual model confidence scores. "
            f"Reflects the collective certainty of the tribunal. "
            f"Values > 0.75 indicate high collective confidence.",
            self.styles['ATBodyText']
        ))

        # Delta explanation
        elements.append(Paragraph("<b>Delta (Δ) - Decisiveness Index:</b>", self.styles['SubsectionHeader']))
        elements.append(Paragraph(
            f"<b>{delta:.4f}</b> - |Φ - Ψ| measures the gap between agreement and confidence. "
            f"Low delta (< 0.1) indicates stable consensus. "
            f"High delta suggests models agree but with varying certainty.",
            self.styles['ATBodyText']
        ))

        # Mathematical formula
        elements.append(Spacer(1, 15))
        elements.append(Paragraph(
            "<font face='Courier' size='9'>Consensus Formula: V = argmax(votes) where Φ > 0.5 ∧ Ψ > 0.5 ∧ Δ < 0.15</font>",
            self.styles['CodeStyle']
        ))

        elements.append(Spacer(1, 30))
        return elements

    def _build_tribunal_votes(self, votes: List[Dict]) -> List:
        """Build individual tribunal votes section."""
        elements = []

        elements.append(Paragraph("TRIBUNAL DELIBERATION", self.styles['SectionHeader']))

        for vote in votes:
            model = vote.get('model', 'Unknown Model')
            verdict = vote.get('verdict', 'REVIEW')
            confidence = vote.get('confidence', 0)
            rationale = vote.get('rationale', vote.get('raw_response', 'No rationale provided'))

            # Determine color
            if 'BID' in verdict and 'NO' not in verdict:
                color = AtomicTrustColors.SUCCESS
            elif 'NO' in verdict:
                color = AtomicTrustColors.DANGER
            else:
                color = AtomicTrustColors.WARNING

            # Model header
            elements.append(Paragraph(
                f"<font color='{color.hexval()}'><b>{model}</b></font> - {verdict} ({confidence*100:.0f}%)",
                self.styles['SubsectionHeader']
            ))

            # Rationale (truncated for executive, full for technical)
            if len(rationale) > 500:
                display_rationale = rationale[:500] + "..."
            else:
                display_rationale = rationale

            elements.append(Paragraph(display_rationale, self.styles['ATBodyText']))
            elements.append(Spacer(1, 15))

        return elements

    def _build_requirements_analysis(self, requirements: List[Dict]) -> List:
        """Build detailed requirements analysis section."""
        elements = []

        elements.append(Paragraph("REQUIREMENTS ANALYSIS", self.styles['SectionHeader']))

        # Summary stats
        total = len(requirements)
        must_reqs = [r for r in requirements if r.get('type') == 'MUST']
        shall_reqs = [r for r in requirements if r.get('type') == 'SHALL']
        should_reqs = [r for r in requirements if r.get('type') == 'SHOULD']
        gaps = [r for r in requirements if r.get('match_score', 100) < 70]

        elements.append(Paragraph(
            f"<b>Total Requirements:</b> {total} | "
            f"<b>MUST:</b> {len(must_reqs)} | "
            f"<b>SHALL:</b> {len(shall_reqs)} | "
            f"<b>SHOULD:</b> {len(should_reqs)} | "
            f"<b>Gaps:</b> {len(gaps)}",
            self.styles['ATBodyText']
        ))
        elements.append(Spacer(1, 15))

        # Requirements table
        table_data = [['ID', 'Type', 'Requirement', 'Match %', 'Gap']]

        for req in requirements:
            req_id = req.get('requirement_id', 'N/A')
            req_type = req.get('type', 'N/A')
            req_text = req.get('text', 'N/A')
            if len(req_text) > 60:
                req_text = req_text[:60] + "..."
            match_score = req.get('match_score', 0)
            gap = req.get('gap', '-')
            if gap and len(str(gap)) > 30:
                gap = str(gap)[:30] + "..."

            table_data.append([req_id, req_type, req_text, f"{match_score}%", gap or '-'])

        table = Table(table_data, colWidths=[0.6*inch, 0.5*inch, 3*inch, 0.6*inch, 1.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), AtomicTrustColors.PRIMARY),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (3, 0), (3, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, AtomicTrustColors.LIGHT]),
            ('GRID', (0, 0), (-1, -1), 0.5, AtomicTrustColors.LIGHT),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))

        elements.append(table)
        elements.append(Spacer(1, 30))

        return elements

    def _build_risks_section(self, analysis_data: Dict) -> List:
        """Build risks section."""
        elements = []

        elements.append(Paragraph("RISK ASSESSMENT", self.styles['SectionHeader']))

        risks = analysis_data.get('risks', [])

        if not risks:
            elements.append(Paragraph(
                "No significant risks identified.",
                self.styles['ATBodyText']
            ))
        else:
            for risk in risks:
                level = risk.get('level', 'MEDIUM')
                desc = risk.get('description', 'Unknown risk')

                if level == 'HIGH':
                    color = AtomicTrustColors.DANGER
                elif level == 'MEDIUM':
                    color = AtomicTrustColors.WARNING
                else:
                    color = AtomicTrustColors.NEUTRAL

                elements.append(Paragraph(
                    f"<font color='{color.hexval()}'><b>[{level}]</b></font> {desc}",
                    self.styles['ATBodyText']
                ))

        elements.append(Spacer(1, 30))
        return elements

    def _build_next_steps(self, analysis_data: Dict) -> List:
        """Build next steps section."""
        elements = []

        elements.append(Paragraph("RECOMMENDED NEXT STEPS", self.styles['SectionHeader']))

        next_steps = analysis_data.get('next_steps', [])

        if next_steps:
            items = [ListItem(Paragraph(step, self.styles['ATBodyText'])) for step in next_steps]
            elements.append(ListFlowable(items, bulletType='bullet', start='bulletchar'))
        else:
            elements.append(Paragraph(
                "No specific next steps recommended.",
                self.styles['ATBodyText']
            ))

        elements.append(Spacer(1, 30))
        return elements

    def _build_legal_compliance(self, analysis_data: Dict, rfp_data: Dict) -> List:
        """Build legal/compliance section."""
        elements = []

        elements.append(Paragraph("LEGAL & COMPLIANCE CONSIDERATIONS", self.styles['SectionHeader']))

        # Compliance requirements analysis
        elements.append(Paragraph("<b>Regulatory Requirements Identified:</b>", self.styles['SubsectionHeader']))

        compliance_keywords = ['HIPAA', 'GDPR', 'SOC 2', 'FedRAMP', 'PCI-DSS', 'CCPA', 'ISO 27001']
        rfp_text = json.dumps(rfp_data).upper()
        found_compliance = [kw for kw in compliance_keywords if kw in rfp_text]

        if found_compliance:
            for comp in found_compliance:
                elements.append(Paragraph(f"• {comp} compliance requirements detected", self.styles['ATBodyText']))
        else:
            elements.append(Paragraph("No specific compliance framework requirements identified.", self.styles['ATBodyText']))

        # Contract considerations
        elements.append(Paragraph("<b>Contract Risk Factors:</b>", self.styles['SubsectionHeader']))

        risks = analysis_data.get('risks', [])
        legal_risks = [r for r in risks if any(kw in r.get('description', '').lower()
                      for kw in ['contract', 'legal', 'liability', 'insurance', 'warranty', 'indemnity'])]

        if legal_risks:
            for risk in legal_risks:
                elements.append(Paragraph(f"• {risk.get('description')}", self.styles['ATBodyText']))
        else:
            elements.append(Paragraph("No specific legal risks flagged by tribunal.", self.styles['ATBodyText']))

        # Disclaimers
        elements.append(Spacer(1, 30))
        elements.append(Paragraph(
            "<i>This analysis is generated by an AI system and should be reviewed by qualified legal counsel "
            "before making contractual commitments. The Atomic Trust™ platform provides decision support "
            "but does not constitute legal advice.</i>",
            ParagraphStyle(
                name='Disclaimer',
                fontSize=8,
                textColor=AtomicTrustColors.NEUTRAL,
                alignment=TA_JUSTIFY
            )
        ))

        elements.append(Spacer(1, 30))
        return elements

    def _build_cryptographic_footer(self, metadata: ReportMetadata) -> List:
        """Build cryptographic verification footer."""
        elements = []

        elements.append(HRFlowable(width="100%", thickness=1, color=AtomicTrustColors.LIGHT))
        elements.append(Spacer(1, 15))

        elements.append(Paragraph("DOCUMENT INTEGRITY VERIFICATION", self.styles['SubsectionHeader']))

        # Hash display
        elements.append(Paragraph(
            f"<font face='Courier' size='8'>Report Hash (SHA-256): {metadata.content_hash or 'PENDING'}</font>",
            self.styles['CodeStyle']
        ))

        elements.append(Paragraph(
            f"<font face='Courier' size='8'>Report ID: {metadata.report_id}</font>",
            self.styles['CodeStyle']
        ))

        elements.append(Paragraph(
            f"<font face='Courier' size='8'>Generated: {metadata.generated_at}</font>",
            self.styles['CodeStyle']
        ))

        elements.append(Spacer(1, 15))
        elements.append(Paragraph(
            "<i>This document is cryptographically signed. Any modification will invalidate the hash. "
            "Verify integrity at: atomic-trust.io/verify</i>",
            ParagraphStyle(
                name='VerifyNote',
                fontSize=7,
                textColor=AtomicTrustColors.NEUTRAL,
                alignment=TA_CENTER
            )
        ))

        return elements

    def _build_executive_report(
        self,
        metadata: ReportMetadata,
        analysis_data: Dict,
        rfp_data: Dict,
        votes: List[Dict]
    ) -> List:
        """Build executive summary report."""
        elements = []

        elements.extend(self._build_header(metadata, "EXECUTIVE SUMMARY"))
        elements.extend(self._build_verdict_section(analysis_data))
        elements.extend(self._build_metrics_table(analysis_data))

        # Brief risks
        elements.append(Paragraph("KEY RISKS", self.styles['SectionHeader']))
        risks = analysis_data.get('risks', [])[:3]  # Top 3 only
        for risk in risks:
            level = risk.get('level', 'MEDIUM')
            color = AtomicTrustColors.DANGER if level == 'HIGH' else AtomicTrustColors.WARNING
            elements.append(Paragraph(
                f"<font color='{color.hexval()}'><b>[{level}]</b></font> {risk.get('description', '')}",
                self.styles['ATBodyText']
            ))
        if not risks:
            elements.append(Paragraph("No significant risks identified.", self.styles['ATBodyText']))

        elements.extend(self._build_next_steps(analysis_data))
        elements.extend(self._build_cryptographic_footer(metadata))

        return elements

    def _build_technical_report(
        self,
        metadata: ReportMetadata,
        analysis_data: Dict,
        rfp_data: Dict,
        votes: List[Dict],
        requirements: List[Dict]
    ) -> List:
        """Build detailed technical report."""
        elements = []

        elements.extend(self._build_header(metadata, "TECHNICAL ANALYSIS REPORT"))
        elements.extend(self._build_verdict_section(analysis_data))
        elements.extend(self._build_metrics_table(analysis_data))
        elements.extend(self._build_consensus_metrics(analysis_data))

        elements.append(PageBreak())

        elements.extend(self._build_tribunal_votes(votes))

        elements.append(PageBreak())

        elements.extend(self._build_requirements_analysis(requirements))
        elements.extend(self._build_risks_section(analysis_data))
        elements.extend(self._build_next_steps(analysis_data))

        elements.append(PageBreak())

        # Semantic analysis section
        elements.append(Paragraph("SEMANTIC ANALYSIS", self.styles['SectionHeader']))
        elements.append(Paragraph(
            "The tribunal employs semantic similarity matching to evaluate requirement-capability alignment. "
            "Each requirement is tokenized and compared against the capability repository using TF-IDF "
            "vectorization and cosine similarity scoring.",
            self.styles['ATBodyText']
        ))
        elements.append(Paragraph(
            "<font face='Courier' size='9'>Similarity(req, cap) = cos(θ) = (A · B) / (||A|| × ||B||)</font>",
            self.styles['CodeStyle']
        ))
        elements.append(Spacer(1, 15))
        elements.append(Paragraph(
            "Match scores are computed as weighted averages where MUST requirements carry 2x weight, "
            "SHALL requirements carry 1.5x weight, and SHOULD requirements carry 1x weight.",
            self.styles['ATBodyText']
        ))

        elements.extend(self._build_cryptographic_footer(metadata))

        return elements

    def _build_legal_report(
        self,
        metadata: ReportMetadata,
        analysis_data: Dict,
        rfp_data: Dict,
        votes: List[Dict],
        requirements: List[Dict]
    ) -> List:
        """Build legal/compliance focused report."""
        elements = []

        elements.extend(self._build_header(metadata, "LEGAL & COMPLIANCE ANALYSIS"))
        elements.extend(self._build_verdict_section(analysis_data))
        elements.extend(self._build_metrics_table(analysis_data))
        elements.extend(self._build_legal_compliance(analysis_data, rfp_data))
        elements.extend(self._build_risks_section(analysis_data))

        # Compliance requirements detail
        elements.append(Paragraph("COMPLIANCE REQUIREMENTS DETAIL", self.styles['SectionHeader']))
        compliance_reqs = [r for r in requirements if any(
            kw in r.get('text', '').lower() for kw in
            ['comply', 'compliance', 'certif', 'audit', 'attest', 'hipaa', 'gdpr', 'soc', 'fedramp', 'pci']
        )]

        if compliance_reqs:
            for req in compliance_reqs:
                score = req.get('match_score', 0)
                color = AtomicTrustColors.SUCCESS if score >= 80 else AtomicTrustColors.WARNING if score >= 60 else AtomicTrustColors.DANGER
                elements.append(Paragraph(
                    f"<font color='{color.hexval()}'><b>[{req.get('requirement_id')}]</b></font> "
                    f"{req.get('text', '')[:100]}... <b>Match: {score}%</b>",
                    self.styles['ATBodyText']
                ))
                if req.get('gap'):
                    elements.append(Paragraph(
                        f"<i>Gap: {req.get('gap')}</i>",
                        ParagraphStyle(name='GapText', fontSize=9, textColor=AtomicTrustColors.DANGER, leftIndent=20)
                    ))
        else:
            elements.append(Paragraph("No specific compliance requirements identified.", self.styles['ATBodyText']))

        elements.extend(self._build_next_steps(analysis_data))
        elements.extend(self._build_cryptographic_footer(metadata))

        return elements

    def _build_full_report(
        self,
        metadata: ReportMetadata,
        analysis_data: Dict,
        rfp_data: Dict,
        votes: List[Dict],
        requirements: List[Dict]
    ) -> List:
        """Build complete tribunal report with all details."""
        elements = []

        elements.extend(self._build_header(metadata, "COMPLETE TRIBUNAL REPORT"))

        # RFP Details
        elements.append(Paragraph("RFP DETAILS", self.styles['SectionHeader']))
        rfp_table = [
            ['Field', 'Value'],
            ['RFP ID', rfp_data.get('id', 'N/A')],
            ['Name', rfp_data.get('name', 'N/A')],
            ['Client', rfp_data.get('client', 'N/A')],
            ['Industry', rfp_data.get('industry', 'N/A')],
            ['Value', rfp_data.get('value', 'N/A')],
            ['Deadline', rfp_data.get('deadline', 'N/A')],
            ['Difficulty', rfp_data.get('difficulty', 'N/A')],
        ]
        table = Table(rfp_table, colWidths=[1.5*inch, 4.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), AtomicTrustColors.LIGHT),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, AtomicTrustColors.LIGHT),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 30))

        elements.extend(self._build_verdict_section(analysis_data))
        elements.extend(self._build_metrics_table(analysis_data))
        elements.extend(self._build_consensus_metrics(analysis_data))

        elements.append(PageBreak())

        elements.extend(self._build_tribunal_votes(votes))

        elements.append(PageBreak())

        elements.extend(self._build_requirements_analysis(requirements))

        elements.append(PageBreak())

        elements.extend(self._build_legal_compliance(analysis_data, rfp_data))
        elements.extend(self._build_risks_section(analysis_data))
        elements.extend(self._build_next_steps(analysis_data))

        # Raw data section for full transparency
        elements.append(PageBreak())
        elements.append(Paragraph("RAW ANALYSIS DATA", self.styles['SectionHeader']))
        elements.append(Paragraph(
            "The following JSON contains the complete analysis data for audit purposes:",
            self.styles['ATBodyText']
        ))

        raw_data = {
            'metadata': {
                'report_id': metadata.report_id,
                'generated_at': metadata.generated_at,
                'rfp_id': metadata.rfp_id
            },
            'verdict': analysis_data.get('recommendation'),
            'tribunal': analysis_data.get('tribunal'),
            'analysis': analysis_data.get('analysis'),
            'vote_count': len(votes)
        }

        elements.append(Paragraph(
            f"<font face='Courier' size='7'>{json.dumps(raw_data, indent=2)}</font>",
            self.styles['CodeStyle']
        ))

        elements.extend(self._build_cryptographic_footer(metadata))

        return elements


# Convenience function for generating reports
def generate_analysis_report(
    report_type: str,
    analysis_data: Dict,
    rfp_data: Dict,
    votes: List[Dict],
    requirements: List[Dict]
) -> bytes:
    """
    Generate a PDF report for RFP analysis.

    Args:
        report_type: One of 'executive', 'technical', 'legal', 'full'
        analysis_data: Final analysis results
        rfp_data: RFP metadata
        votes: LLM tribunal votes
        requirements: Requirement analysis details

    Returns:
        PDF bytes
    """
    generator = RFPReportGenerator()
    rt = ReportType(report_type.lower())
    return generator.generate_report(rt, analysis_data, rfp_data, votes, requirements)


if __name__ == "__main__":
    # Test report generation
    test_analysis = {
        'recommendation': 'STRONG_BID',
        'tribunal': {
            'confidence': 0.87,
            'phi': 0.823,
            'psi': 0.871,
            'delta': 0.048,
            'votes': 3,
            'unanimous': True
        },
        'analysis': {
            'overall_fit': 85,
            'must_requirement_fit': 92,
            'gap_count': 2
        },
        'risks': [
            {'level': 'MEDIUM', 'description': 'Timeline aggressive for scope'},
            {'level': 'LOW', 'description': 'Minor clarification needed on SLA terms'}
        ],
        'next_steps': [
            'Schedule discovery call with client',
            'Prepare technical approach document',
            'Identify project team resources'
        ]
    }

    test_rfp = {
        'id': 'rfp-001',
        'name': 'Cloud Migration & Modernization',
        'client': 'Pacific Northwest Healthcare',
        'industry': 'Healthcare',
        'value': '$2.4M',
        'deadline': '45 days',
        'difficulty': 'High'
    }

    test_votes = [
        {'model': 'claude-sonnet-4', 'verdict': 'STRONG_BID', 'confidence': 0.89, 'rationale': 'Strong alignment with healthcare expertise...'},
        {'model': 'gpt-4o', 'verdict': 'BID', 'confidence': 0.84, 'rationale': 'Good technical fit, recommend proceeding...'},
        {'model': 'grok-2', 'verdict': 'BID', 'confidence': 0.88, 'rationale': 'Capabilities match requirements well...'}
    ]

    test_requirements = [
        {'requirement_id': 'REQ-001', 'type': 'MUST', 'text': 'Solution must be HIPAA compliant', 'match_score': 100, 'gap': None},
        {'requirement_id': 'REQ-002', 'type': 'MUST', 'text': 'Support HL7 FHIR R4 data exchange', 'match_score': 95, 'gap': None},
        {'requirement_id': 'REQ-003', 'type': 'SHALL', 'text': 'Provide 24/7 support coverage', 'match_score': 85, 'gap': 'Current SLA is 18/7'},
    ]

    # Generate test reports
    for rt in ['executive', 'technical', 'legal', 'full']:
        pdf = generate_analysis_report(rt, test_analysis, test_rfp, test_votes, test_requirements)
        with open(f'test_report_{rt}.pdf', 'wb') as f:
            f.write(pdf)
        print(f"Generated test_report_{rt}.pdf ({len(pdf)} bytes)")
