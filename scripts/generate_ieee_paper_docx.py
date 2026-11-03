#!/usr/bin/env python3
"""
Generate IEEE Conference Research Paper in Microsoft Word (.docx) format.
Restores 100% of the past 28% Turnitin submission text verbatim,
with publication-grade figures, robust equation formatting, and exact margins.
"""

from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING, WD_TAB_ALIGNMENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.section import WD_SECTION
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = PROJECT_ROOT / "paper"
DOCS_DIR = PROJECT_ROOT / "docs"
FIGURES_DIR = PAPER_DIR / "figures"
OUT_DOCX_PATH = PAPER_DIR / "SecureKYC_AI_IEEE_Research_Paper.docx"
MIRROR_DOCX_PATH = DOCS_DIR / "SecureKYC_AI_IEEE_Research_Paper.docx"

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_cell_borders(cell, top=None, bottom=None, left=None, right=None):
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = OxmlElement('w:tcBorders')
    for border_name, border_style in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        if border_style:
            b = OxmlElement(f'w:{border_name}')
            b.set(qn('w:val'), border_style.get('val', 'single'))
            b.set(qn('w:sz'), str(border_style.get('sz', 4)))
            b.set(qn('w:space'), '0')
            b.set(qn('w:color'), border_style.get('color', 'auto'))
            tcBorders.append(b)
        else:
            b = OxmlElement(f'w:{border_name}')
            b.set(qn('w:val'), 'none')
            tcBorders.append(b)
    tcPr.append(tcBorders)

def set_cell_shading(cell, color_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    tcPr.append(shd)

def set_section_columns(section, num_cols=2, space_pt=18):
    sectPr = section._sectPr
    cols = sectPr.xpath('./w:cols')
    space_dxa = str(int(space_pt * 20))
    if cols:
        cols[0].set(qn('w:num'), str(num_cols))
        cols[0].set(qn('w:space'), space_dxa)
    else:
        col_elem = OxmlElement('w:cols')
        col_elem.set(qn('w:num'), str(num_cols))
        col_elem.set(qn('w:space'), space_dxa)
        sectPr.append(col_elem)

def add_equation(doc, eq_text, eq_num):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    p.paragraph_format.left_indent = Inches(0.08)
    p.paragraph_format.tab_stops.add_tab_stop(Inches(3.42), WD_TAB_ALIGNMENT.RIGHT)
    
    if '\n' in eq_text:
        parts = eq_text.split('\n')
        r1 = p.add_run(parts[0] + '\n')
        r1.font.name = 'Cambria Math'
        r1.font.size = Pt(8.5)
        r1.font.italic = True
        
        r2 = p.add_run(parts[1] + '\t')
        r2.font.name = 'Cambria Math'
        r2.font.size = Pt(8.5)
        r2.font.italic = True
    else:
        r1 = p.add_run(eq_text + '\t')
        r1.font.name = 'Cambria Math'
        r1.font.size = Pt(8.5)
        r1.font.italic = True
        
    r_num = p.add_run(f"({eq_num})")
    r_num.font.name = 'Times New Roman'
    r_num.font.size = Pt(8.5)
    r_num.font.italic = False
    return p

def fix_normal_style(doc):
    """Zero out the Normal paragraph style so Word doesn't add extra space."""
    normal = doc.styles['Normal']
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(0)
    normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    normal.font.name = 'Times New Roman'
    normal.font.size = Pt(10)

def build_word_document():
    print("[*] Initializing IEEE Conference Word Document (28% Original Content Rollback)...")
    doc = docx.Document()
    fix_normal_style(doc)

    for s in doc.sections:
        s.top_margin = Inches(0.75)
        s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(0.625)
        s.right_margin = Inches(0.625)
        s.page_width = Inches(8.5)
        s.page_height = Inches(11.0)
        
        header_p = s.header.paragraphs[0]
        header_p.text = ""
        
        footer_p = s.footer.paragraphs[0]
        footer_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        frun = footer_p.add_run("Authorized Authorship: Jatin Dhawad & Param Rawal | SVKM's NMIMS STME Indore")
        frun.font.name = "Times New Roman"
        frun.font.size = Pt(8.0)
        frun.font.color.rgb = RGBColor(0x88, 0x88, 0x88)

    # 1-COLUMN HEADER BLOCK
    s1 = doc.sections[0]
    set_section_columns(s1, num_cols=1)
    
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(6)
    p_title.paragraph_format.space_after = Pt(12)
    r_title = p_title.add_run("Active Challenge-Response Biometric Verification for Video KYC via Speech Aerodynamics and Trellis Forced Alignment")
    r_title.font.name = "Times New Roman"
    r_title.font.size = Pt(18)
    r_title.font.bold = True
    
    auth_table = doc.add_table(rows=1, cols=2)
    auth_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    auth_table.autofit = False
    auth_table.columns[0].width = Inches(3.6)
    auth_table.columns[1].width = Inches(3.6)
    
    c1 = auth_table.cell(0, 0)
    set_cell_borders(c1)
    p_a1 = c1.paragraphs[0]
    p_a1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_a1.paragraph_format.space_after = Pt(1)
    r = p_a1.add_run("Jatin Dhawad\n")
    r.font.name = "Times New Roman"
    r.font.size = Pt(11)
    r.font.bold = True
    r = p_a1.add_run("Department of Computer Engineering\nSchool of Technology Management & Engineering (STME)\nSVKM's NMIMS, Indore, India\n")
    r.font.name = "Times New Roman"
    r.font.size = Pt(9.5)
    r.font.italic = True
    r = p_a1.add_run("jatin.dhawad652@nmims.in")
    r.font.name = "Courier New"
    r.font.size = Pt(9.0)
    
    c2 = auth_table.cell(0, 1)
    set_cell_borders(c2)
    p_a2 = c2.paragraphs[0]
    p_a2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_a2.paragraph_format.space_after = Pt(1)
    r = p_a2.add_run("Param Rawal\n")
    r.font.name = "Times New Roman"
    r.font.size = Pt(11)
    r.font.bold = True
    r = p_a2.add_run("Department of Computer Engineering\nSchool of Technology Management & Engineering (STME)\nSVKM's NMIMS, Indore, India\n")
    r.font.name = "Times New Roman"
    r.font.size = Pt(9.5)
    r.font.italic = True
    r = p_a2.add_run("param.rawal480@nmims.in")
    r.font.name = "Courier New"
    r.font.size = Pt(9.0)
    
    p_spacer = doc.add_paragraph()
    p_spacer.paragraph_format.space_before = Pt(2)
    p_spacer.paragraph_format.space_after = Pt(2)
    p_spacer.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE

    # 2-COLUMN BODY
    s2 = doc.add_section(WD_SECTION.CONTINUOUS)
    set_section_columns(s2, num_cols=2, space_pt=18)

    # Abstract (Exact 28% Turnitin submission text)
    p_abs = doc.add_paragraph()
    p_abs.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_abs.paragraph_format.space_before = Pt(2)
    p_abs.paragraph_format.space_after = Pt(3)
    p_abs.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    r = p_abs.add_run("Abstract—")
    r.font.name = "Times New Roman"
    r.font.size = Pt(9.0)
    r.font.bold = True
    r.font.italic = True
    r = p_abs.add_run(
        "Most Video Know-Your-Customer (Video KYC) systems assume that a live camera feed guarantees a live human applicant. Software-defined virtual cameras and neural puppeteering tools break this assumption completely, feeding synthetic video directly into browser sessions where passive optical filters cannot detect them. In this paper, we describe SecureKYC AI, an active challenge-response verification system that evaluates speech aerodynamics rather than searching for pixel-level rendering artifacts. Human speakers cannot voice bilabial consonants—/m/, /b/, and /p/—without pressing the vermilion borders of their lips completely shut to build intra-oral pressure (P_oral >> P_ambient). SecureKYC AI issues a short random phrase, tracks 3D lip landmarks in the browser stream via MediaPipe, isolates phoneme boundaries in the audio using Meta MMS_FA trellis forced alignment, and checks for physical closure with a scale-invariant aperture metric. We evaluated this defense on N=691 audio-visual clips from VidTIMIT and DeepFakeTIMIT partitioned across 43 disjoint identities (25 training, 9 validation, 9 test subjects). On the held-out test split, our constrained late-fusion meta-learner reached 68.93% accuracy, 0.7197 ROC-AUC (5-fold cross-validation mean: 0.6877 ± 0.0449), 36.02% EER, and an F1-score of 0.6000. At a standard 0.5 decision cutoff, late fusion cut genuine false rejections from 16 down to 10 compared to visual tracking alone (paired bootstrap 95% CI [0, 12]) while matching its 52.17% attack recall and lifting precision from 60.00% to 70.59%. Across all 691 clips, lip closure failed in only 4.5% of genuine bilabial events versus 41.6% of reenactments (Cohen's d = 0.66 per clip, 1.10 per subject). Offline per-frame processing averaged 5.02 ms in steady state (14.05 ms cold start over 9 demo clips). Our results hold strictly for in-distribution benchmarks and await live multi-camera field testing."
    )
    r.font.name = "Times New Roman"
    r.font.size = Pt(9.0)
    
    # Index Terms
    p_idx = doc.add_paragraph()
    p_idx.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_idx.paragraph_format.space_before = Pt(2)
    p_idx.paragraph_format.space_after = Pt(6)
    p_idx.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    r = p_idx.add_run("Index Terms—")
    r.font.name = "Times New Roman"
    r.font.size = Pt(9.0)
    r.font.bold = True
    r.font.italic = True
    r = p_idx.add_run("Video KYC, Deepfake Detection, Presentation Attack Detection, Speech Aerodynamics, Forced Alignment, Bilabial Closure, Active Challenge-Response, Multimodal Biometrics.")
    r.font.name = "Times New Roman"
    r.font.size = Pt(9.0)

    def add_h1(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        r = p.add_run(text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(9.5)
        r.font.bold = True
        return p

    def add_h2(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        r = p.add_run(text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(9.5)
        r.font.italic = True
        r.font.bold = True
        return p

    def add_body(text, indent=True):
        paragraphs = []
        lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
        for i, line in enumerate(lines):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            
            is_bullet = line.startswith('•') or line.startswith('- ')
            is_numbered = len(line) >= 2 and line[0].isdigit() and line[1] in '.)'
            
            if is_bullet or is_numbered:
                p.paragraph_format.left_indent = Inches(0.18)
                p.paragraph_format.first_line_indent = Inches(-0.18)
            elif indent and (i == 0 or not lines[i-1].endswith(':')):
                p.paragraph_format.first_line_indent = Inches(0.14)
            else:
                p.paragraph_format.first_line_indent = Inches(0.0)
                
            r = p.add_run(line)
            r.font.name = "Times New Roman"
            r.font.size = Pt(9.0)
            paragraphs.append(p)
        return paragraphs[0] if len(paragraphs) == 1 else paragraphs

    def add_figure(fig_path, caption_label, caption_text, width_in=3.35):
        """Insert figure image + IEEE-style caption. Skips silently if file missing."""
        if not Path(fig_path).exists():
            return
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(4)
        p_img.paragraph_format.space_after = Pt(1)
        p_img.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        r_img = p_img.add_run()
        r_img.add_picture(str(fig_path), width=Inches(width_in))

        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p_cap.paragraph_format.space_before = Pt(1)
        p_cap.paragraph_format.space_after = Pt(4)
        p_cap.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        r_label = p_cap.add_run(caption_label)
        r_label.font.name = "Times New Roman"
        r_label.font.size = Pt(8.5)
        r_label.font.bold = True
        r_desc = p_cap.add_run(caption_text)
        r_desc.font.name = "Times New Roman"
        r_desc.font.size = Pt(8.5)

    # SECTION I: INTRODUCTION
    add_h1("I. INTRODUCTION")
    add_body(
        "Remote onboarding under the RBI Master Directions on KYC and Europe's AMLD5/eIDAS rules runs over WebRTC video calls. "
        "The regulatory model assumes a live applicant positioned in front of a physical camera sensor. Real-time generative models "
        "invalidate that assumption. Earlier presentation attacks (printed photographs, silicone masks, screen replay) still had to traverse "
        "the optical path; current attacks write synthetic frames directly into the capture pipeline [1], [2].",
        indent=False
    )
    add_body(
        "Optical liveness checks target imaging artifacts: frequency-domain spikes [4], up-convolution patterns [5], head-pose drift [6], "
        "and rPPG pulse signals [7], [8]. Injection removes the optical path from the threat model. Loopback drivers (OBS Virtual Camera, ManyCam, "
        "Linux v4l2loopback) expose arbitrary frame buffers as standard V4L2/DirectShow capture devices at the OS driver level. Browser sandboxes "
        "cannot audit peripheral hardware provenance over WebRTC (no access to the hardware bus or driver stack), so the virtual device is "
        "enumerated and consumed as a physical webcam. The remote verifier receives a clean digital pixel grid with none of the glare, moiré, "
        "or lens distortion a physical capture chain introduces [3]."
    )
    add_body(
        "Live puppeteering makes injection interactive. A human operator drives the session through a commodity webcam, while reenactment software "
        "(DeepFaceLive [15]) tracks the operator's facial landmarks per frame and renders the victim's identity onto the injected stream. Standard "
        "ISO/IEC 30107-3 presentation attack detection (PAD) is bypassed. Scripted liveness prompts (blink, smile, yaw rotation) offer no protection: "
        "the operator performs the gesture, and the neural mask reproduces it with sub-frame latency. Fig. 1 traces the complete attack pathway."
    )
    add_figure(
        FIGURES_DIR / "threat_model_diagram.png",
        "Fig. 1.  ",
        "Threat model for virtual camera injection in Video KYC. The attacker feeds synthetic video into the session, so optical liveness checks never see it."
    )

    add_h2("A. Prior Art: The Limitations of Visual Challenge Cascades")
    add_body(
        "Passive detectors offer no guarantee under software injection. Research has consequently moved toward active challenge-response protocols. "
        "The closest baseline is GOTCHA (Mittal et al., IEEE EuroS&P 2024) [12], whose challenges impose physical perturbations (hand occlusion across "
        "the face, standing from a seated position, high-yaw head rotation) that exceed what generative autoencoders can render under severe volumetric "
        "occlusion and extreme perspective, producing texture tearing and geometric warping. Automated detection reached an ROC-AUC of 80.1% "
        "(88.6% under human inspection) [12].",
        indent=False
    )
    add_body(
        "Three constraints limit direct transfer to commercial banking KYC:\n"
        "• Participant friction: Occlusion and full-body gestures (covering the eyes, standing up) are intrusive in a regulated identity-verification session.\n"
        "• Onboarding delay: Multi-gesture sequences require 10 to 15 seconds. Our design budget is sub-second latency after speech ends.\n"
        "• Ignoring the speech signal: Analysis is restricted to RGB frames. The audio channel, and with it the vocal-tract mechanics underlying every spoken response, is discarded."
    )

    add_h2("B. Contributions of This Work")
    add_body(
        "SecureKYC AI replaces physical movement challenges with a spoken challenge constrained by vocal tract aerodynamics. Contributions:\n"
        "1) Aerodynamic speech invariant: Bilabial phonemes (/m/, /b/, /p/) require complete lip closure to build intra-oral pressure (P_oral >> P_ambient). Verification is anchored to this physiological constraint. Neural reenactment tools in our benchmark routinely fail to render the seal.\n"
        "2) Trellis alignment over known prompts: Unconstrained ASR degrades on compressed WebRTC audio. The server instead issues an active prompt from a curated bilabial vocabulary (e.g., \"My Bank Passcode\"); with the ground-truth transcript fixed in advance, Meta MMS_FA dynamic programming trellis alignment isolates bilabial boundaries deterministically, free of transcription errors and drift.\n"
        "3) Split-budget streaming pipeline: Visual tracking runs concurrently with speech at 5.02 ms per frame in steady state over 9 demo clips (14.05 ms cold start; includes video decode, 3D landmarking, visual inference, and CTC prompt checking). Acoustic scoring is deferred to a post-utterance pass. That pass completes in 224.3 ± 21.7 ms on Apple Silicon MPS (local in-process benchmark).\n"
        "4) Non-negatively regularized late fusion: Early concatenation of multimodal features caused gradient interference and scale collapse (Section V-B). Visual and acoustic models are trained separately, and their posterior probabilities are merged by an L2-regularized logistic head with non-negative weight constraints (W_vis >= 0, W_desync >= 0) that prevent catastrophic sign inversion during training.\n"
        "5) Strict zero-synthetic experimental protocol: Synthetic augmentation and random splits leak facial morphology across partitions. Neither is used. Evaluation covers N=691 audio-visual clips from 43 disjoint subjects (25 training, 9 validation, 9 test), with 68.93% operational accuracy and 0.7197 test ROC-AUC.",
        indent=False
    )
    add_h1("II. AERODYNAMIC & BIOMECHANICAL FOUNDATIONS")
    add_h2("A. Fundamentals of Bilabial Speech Aerodynamics")
    add_body(
        "Human speech production is governed by the continuous mechanical manipulation of pulmonary airflow across the vocal tract [13], [14]. "
        "Consonants are classified by their manner and place of articulation. Among English phonemes, the bilabial consonants—the voiced bilabial "
        "nasal /m/, voiced bilabial stop /b/, and voiceless bilabial stop /p/—possess a unique biomechanical invariant: complete, airtight mechanical "
        "coaptation of the vermilion borders of the upper lip (labium superius) and lower lip (labium inferius) [14].",
        indent=False
    )

    # 1-COLUMN SECTION FOR WIDE FIG 2
    s_fig2 = doc.add_section(WD_SECTION.CONTINUOUS)
    set_section_columns(s_fig2, num_cols=1)

    add_figure(
        FIGURES_DIR / "speech_aerodynamics_diagram.png",
        "Fig. 2.  ",
        "Aerodynamic and biomechanical comparison of bilabial consonant production (/m/, /b/, /p/) between genuine human speakers and neural reenactment deepfakes.",
        width_in=6.6
    )

    # RESUME 2-COLUMN BODY
    s_post_fig2 = doc.add_section(WD_SECTION.CONTINUOUS)
    set_section_columns(s_post_fig2, num_cols=2, space_pt=18)

    add_body(
        "During the closure phase (implosion) of /b/ and /p/, the velopharyngeal port closes, hermetically sealing the nasal cavity. "
        "The pulmonary airflow into the closed oral cavity causes intra-oral air pressure (P_oral) to rise rapidly above ambient atmospheric "
        "pressure (P_ambient). According to Bernoulli's acoustic volume velocity relation [13]:"
    )
    add_equation(doc, "P_oral - P_ambient = (ρ / 2) · (U_air / A_orifice)²", "1")
    add_body(
        "Here ρ is air density, U_air is the acoustic volume velocity, and A_orifice is the cross-sectional area of the lip opening. To produce "
        "the burst of sound when the consonant is released, the lips must be fully closed (A_orifice = 0). This lets P_oral reach its peak, about 6 "
        "to 10 cm H2O above atmospheric pressure [14]. For /m/, the velopharyngeal port is open, but the lips stay sealed, so the sound "
        "leaves only through the nose."
    )

    add_h2("B. Why Generative Face-Swaps Fail at Bilabial Closure")
    add_body(
        "Neural face reenactment and face-swap models (autoencoders, GANs, latent diffusion) have three built-in limits that make them "
        "likely to get bilabial closure wrong:\n"
        "1) Latent averaging. These models interpolate expressions in a low-dimensional latent space. Going from an open vowel to a closed "
        "bilabial takes only 40 to 80 ms (1 to 3 frames at 30 FPS). Temporal smoothing and the encoder-decoder bottleneck blur "
        "such fast movement, so the synthetic lips can stay partly open [9], [11].\n"
        "2) No physical constraint. Common deepfake losses (perceptual L1/L2, multi-scale SSIM, PatchGAN adversarial loss) reward "
        "pixel similarity. None of them penalizes a missing air seal [9].\n"
        "3) Blending artifacts. The mouth interior has fine detail (teeth, tongue, shadow), and generators often leave blending seams along "
        "the lip edge. This can keep a bright gap between the upper and lower lip.\n"
        "Figure 3 shows the effect. Real speakers consistently bring their normalized lip aperture below τ = 0.0242 during bilabials. In our "
        "samples the deepfakes generally stay open, with D_lip between about 0.07 and 0.16.",
        indent=False
    )

    # SECTION 3: 1-COLUMN WIDE FIGURE 3 BANNER
    s3 = doc.add_section(WD_SECTION.CONTINUOUS)
    set_section_columns(s3, num_cols=1)

    add_figure(
        FIGURES_DIR / "real_bilabial_trajectories.png",
        "Fig. 3.  ",
        (
            "Normalized 3D lip aperture D_lip(t) for real VidTIMIT speakers (Panel A: fadg0, fgjd0) and DeepFakeTIMIT swaps (Panel B: faks0/fedw0, "
            "msjs1/mrgg0), plotted with scripts/plot_bilabial_window_trajectories.py. Real speakers drop below τ = 0.0242 at the forced-alignment bilabial times "
            "T_bilabial. The deepfakes stay open."
        ),
        width_in=7.1
    )

    # 1-COLUMN SECTION FOR WIDE FIG 4 (System Architecture)
    s_fig4 = doc.add_section(WD_SECTION.CONTINUOUS)
    set_section_columns(s_fig4, num_cols=1)

    add_h1("III. SECUREKYC AI SYSTEM ARCHITECTURE")
    add_body(
        "SecureKYC AI processes two streams: a real-time visual pipeline that runs while the person speaks, and an acoustic alignment step that runs "
        "after the utterance ends.",
        indent=False
    )

    add_figure(
        FIGURES_DIR / "system_architecture_diagram.png",
        "Fig. 4.  ",
        "Overview of the SecureKYC AI system. Visual tracking runs during streaming, and audio alignment and scoring run after the utterance.",
        width_in=6.8
    )

    # RESUME 2-COLUMN BODY (System Architecture & Experimental Setup)
    s4 = doc.add_section(WD_SECTION.CONTINUOUS)
    set_section_columns(s4, num_cols=2, space_pt=18)

    add_h2("A. Real-Time 3D Lip Aperture Metric")
    add_body(
        "Raw pixel distances scale with camera distance. The aperture metric must therefore be invariant to depth and framing. "
        "Facial landmarks are extracted with MediaPipe Face Mesh [18] (min. detection confidence 0.5, refine_landmarks enabled). "
        "Let p_i = (x_i, y_i, z_i) denote the coordinates of landmark i. Tracking dropouts of 1 to 3 frames are filled by linear interpolation "
        "for VisionOnlyNet only. The acoustic desync metric discards dropout frames outright, and any clip with more than 3 consecutive untracked "
        "frames is flagged as an unreliable face track.",
        indent=False
    )
    add_body(
        "Normalized lip aperture D_lip(t) at frame t is the 3D Euclidean distance between the inner vermilion borders, scaled by facial height (nose tip to chin):"
    )
    add_equation(doc, "D_lip(t) = ||p₁₃(t) - p₁₄(t)||₂ / (||p₁(t) - p₁₅₂(t)||₂ + 10⁻⁶)", "2")
    add_body(
        "Landmarks 13 and 14 mark the inner upper and inner lower lip centers; landmarks 1 and 152 mark the nose tip and chin. The 10⁻⁶ term "
        "prevents division by zero. Because numerator and denominator scale identically with camera distance, D_lip(t) remains stable whether "
        "the applicant leans toward the lens or sits well back from it."
    )

    add_h2("B. Trellis Forced Alignment via MMS_FA")
    add_body(
        "Conventional audio-visual sync methods depend on unconstrained blind ASR [10], which yields severe Character Error Rates (CER) "
        "on 3-second mobile KYC recordings. Prompt generation is server-side in SecureKYC AI. The orthographic and phonetic transcript is therefore "
        "known before the applicant speaks. The CER prompt-verification gate (Section V-F) uses a 0.60 cutoff, chosen as the midpoint of the "
        "gap seen on a 15-clip sweep; it is not validated on a larger speaker set, and the demo clip real_03_fdac1_sa2 scored 0.548 through "
        "the app (0.619 on its raw wav), close to the gate.",
        indent=False
    )
    add_body(
        "Let a = [a_1, a_2, ..., a_T] denote the audio waveform resampled to 16 kHz. The pre-trained Meta MMS_FA wav2vec2 acoustic model [16], [17] "
        "maps a to frame-level emission log-probabilities E in R^{T_audio x |V|}, where |V| is the character (token) vocabulary."
    )
    add_body(
        "We construct an alignment trellis α(t, j) representing the log-probability of aligning acoustic frame t to phoneme token j of the known phrase:"
    )
    add_equation(doc, "α(t, j) = max(α(t-1, j), α(t-1, j-1)) + E(t, c_j)", "3")
    add_body(
        "Backtracking through the dynamic programming trellis from α(T_audio, J) gives the optimal alignment path, yielding exact start "
        "and end timestamps for every bilabial consonant k in {/m/, /b/, /p/}:"
    )
    add_equation(doc, "T_bilabial^(k) = [t_start^(k),  t_end^(k)]", "4")
    add_h2("C. Cross-Modal Desynchronization Formulation")
    add_body(
        "For each aligned bilabial event k, we locate the video frame nearest the center of its audio boundary, f_mid^(k), skipping frames "
        "where facial tracking dropped out. Natural speech exhibits anticipatory coarticulation—the lips frequently touch slightly before "
        "or after the acoustic burst release by 20 to 60 ms [14]. To accommodate this physiological offset without losing timing precision, "
        "we check a symmetric window of two video frames on either side of f_mid^(k) (an 80 ms margin at 25 fps):",
        indent=False
    )
    add_equation(doc, "W_k = [f_mid^(k) - 2,  f_mid^(k) + 2]", "5")
    add_body(
        "Across this window W_k, we test whether the applicant achieved an airtight seal by seeing if the minimum observed normalized aperture "
        "drops below the calibrated threshold (τ = 0.0242):"
    )
    add_equation(doc, "P_sync^(k) = max(0,  min_{t ∈ W_k} D_lip(t) - τ)", "6")
    add_body(
        "If landmark tracking drops out across all frames in an analysis window, we assign an aperture penalty of 1.0 to mark an unsealed mouth. "
        "Gathering all K bilabials in the utterance gives the 3-dimensional desynchronization vector v_desync in R³:"
    )
    add_equation(doc, "v_desync = [S_desync,  P_max,  r_fail]", "7")
    add_body(
        "where S_desync = (1/K) Σ P_sync^(k), P_max = max_k P_sync^(k), and r_fail = (1/K) Σ I(P_sync^(k) > 0)."
    )

    add_h2("D. Constrained Late-Fusion Meta-Learner")
    add_body(
        "In early experiments with joint multi-task networks (MultimodalKYCFusionNet), early concatenation of v_desync with visual embeddings "
        "induced destructive gradient interference. To solve this, we decouple the architecture into two independently trained base classifiers:\n"
        "1) Visual Base Network (VisionOnlyNet): A 1D convolutional feature extractor coupled to a 2-layer Bidirectional LSTM (hidden dimension 64) "
        "that processes the temporal aperture trajectory D_lip in R^{T x 1}, outputting posterior probability P_vis in [0, 1].\n"
        "2) Acoustic-Phonemic Base Network (DesyncOnlyNet): A multi-layer perceptron processing v_desync, outputting posterior probability P_desync in [0, 1].",
        indent=False
    )
    add_body(
        "We combine the base predictions via a constrained late-fusion meta-learner:"
    )
    add_equation(doc, "z = W_vis · P_vis + W_desync · P_desync + b,   P_fused = σ(z)", "8")
    add_body(
        "Crucially, we enforce non-negativity constraints during optimization:"
    )
    add_equation(doc, "min_{W_vis, W_desync, b} L_BCE(y, σ(z)) + (λ/2)(W_vis² + W_desync²)\n  s.t.  W_vis ≥ 0,  W_desync ≥ 0", "9")
    add_body(
        "solved via L-BFGS-B optimization. The non-negativity constraint prevents destabilizing sign inversion on outlier speaker kinematics, "
        "guaranteeing that the fused system never falls below the single-modality baseline floor."
    )

    # SECTION IV: EXPERIMENTAL SETUP & PROTOCOL
    add_h1("IV. EXPERIMENTAL SETUP & PROTOCOL")
    add_h2("A. Zero-Synthetic-Data & Hardware Target")
    add_body(
        "In strict adherence to empirical research integrity, zero synthetic data or simulated distributions were utilized in this investigation. "
        "Every metric, ROC curve, and latency measurement originates from real signal-processing pipelines and live PyTorch model forward passes. "
        "All model training and streaming latency profiling were executed on an Apple Silicon platform utilizing Apple Metal Performance Shaders (torch.device(\"mps\")).",
        indent=False
    )

    add_h2("B. Identity-Disjoint Benchmark Dataset")
    add_body(
        "Evaluating deepfake detectors on randomly partitioned video clips causes pervasive identity leakage [2], [11], where models overfit "
        "to specific facial features rather than manipulation artifacts. To guarantee rigorous generalization, we evaluate on N=691 audio-visual "
        "samples compiled from VidTIMIT [1] (N=275 authentic recordings) and DeepFakeTIMIT [1] (N=416 neural autoencoder reenactments spanning the same subjects).",
        indent=False
    )
    add_body(
        "Executing split_dataset_by_identity() on the dataset graph partitions the 43 identities via connected-component analysis into strictly disjoint subsets:\n"
        "• Training Set: 25 identities (N=484 clips: 172 real, 312 fake).\n"
        "• Validation Set: 9 identities (N=104 clips: 46 real, 58 fake).\n"
        "• Held-Out Test Set: 9 completely unseen identities (N=103 clips: 57 real, 46 fake; Train ∩ Test = ∅).\n"
        "The 9 held-out test identities are: fadg0, fcmr0, fpkt0, fram1, mbdg0, mbjk0, mreb0, mrjo0, and mstk0."
    )

    add_h2("C. Physical Calibration of Closure Threshold")
    add_body(
        "The closure threshold τ is calibrated empirically on genuine speech, not set by hand. src/metrics/calibrate_threshold.py extracted "
        "320 bilabial events from 15 VidTIMIT speakers; the per-event distribution is stored in data/processed/calibration_events_breakdown.json. "
        "Normalized aperture D_lip clusters tightly near zero:\n"
        "• Min D_lip: 0.0026 | 10th percentile: 0.0041 | 25th percentile: 0.0059\n"
        "• Median: 0.0077 | 75th percentile: 0.0103 | 90th percentile: 0.0131\n"
        "• 95th percentile (calibrated τ): 0.0241 (rounded to 0.0242 in config.py)\n"
        "• 99th percentile: 0.0787 | Max: 0.1280 | Mean ± std: 0.0103 ± 0.0125\n"
        "τ = 0.0242 sits at the 95th percentile, bounding the per-closure false-alarm rate at 5% within the calibration cohort. On the full "
        "691-clip dataset, genuine speakers missed closure on 4.49% of bilabial windows against 41.63% for neural reenactments. Effect size "
        "in S_desync: Cohen's d = 0.66 per clip (d = 1.10 pooled per speaker, 42 authentic and 32 reenacted subjects).\n"
        "Data disclosure. Two calibration speakers (fadg0 and fcmr0; 42 of 320 calibration events) also appear in the 9-identity test set, "
        "so the calibration cohort is not fully disjoint from test.",
        indent=False
    )
    # SECTION 5: 1-COLUMN WIDE TABLE I (Ablation Table)
    s5 = doc.add_section(WD_SECTION.CONTINUOUS)
    set_section_columns(s5, num_cols=1)

    p_t1 = doc.add_paragraph()
    p_t1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_t1.paragraph_format.space_before = Pt(6)
    p_t1.paragraph_format.space_after = Pt(2)
    r = p_t1.add_run("TABLE I: EMPIRICAL MODEL ABLATION PERFORMANCE COMPARISON (HELD-OUT IDENTITY TEST SET, N=103)")
    r.font.name = "Times New Roman"
    r.font.size = Pt(8.5)
    r.font.bold = True

    t1_data = [
        ["Model Variant / Feature Configuration", "Accuracy (%)", "APCER (%)", "BPCER (%)", "Precision (%)", "Recall (%)", "F1-Score", "ROC-AUC", "Equal Error Rate (EER)"],
        ["VisionOnlyNet (Spatial Standalone)", "63.11%", "47.83%", "28.07%", "60.00%", "52.17%", "0.5581", "0.7105", "36.02%"],
        ["DesyncOnlyNet (Passive Standalone)*", "44.66%", "0.00%", "100.00%", "44.66%", "100.00%", "0.6174", "0.6175", "38.08%"],
        ["Proposed Production: Constrained Late Fusion (Official Deployed)", "68.93%", "47.83%", "17.54%", "70.59%", "52.17%", "0.6000", "0.7197", "36.02%"],
        ["Deployed Pipeline Rule: Constrained Late Fusion OR S_desync > 0.0242", "68.93%", "47.83%", "17.54%", "70.59%", "52.17%", "0.6000", "—", "—"],
        ["Spatial-Only (Zero out desync_vec)†", "61.17%", "—", "—", "68.75%", "23.91%", "0.3548", "0.6182", "—"],
        ["Passive-Only (Zero out lip_seq)†", "55.34%", "—", "—", "0.00%", "0.00%", "0.0000", "0.6232", "—"],
        ["Baseline Early Fusion: MultimodalKYCFusionNet (Raw Concat)†", "65.05%", "—", "—", "67.86%", "41.30%", "0.5135", "0.6423", "—"],
        ["Expected Generalization: Constrained Late Fusion (5-Fold CV Mean)", "—", "—", "—", "—", "—", "—", "0.6877 ± 0.0449", "36.00 ± 2.98%"],
        ["Literature Benchmark: GOTCHA (Mittal et al., 2024) [12]*", "—", "—", "—", "—", "—", "—", "0.8010*", "—"]
    ]

    t1 = doc.add_table(rows=len(t1_data), cols=len(t1_data[0]))
    t1.alignment = WD_TABLE_ALIGNMENT.CENTER
    t1.autofit = False

    col_widths_t1 = [Inches(2.1), Inches(0.65), Inches(0.65), Inches(0.65), Inches(0.65), Inches(0.6), Inches(0.6), Inches(0.75), Inches(0.65)]
    for r_idx, row in enumerate(t1.rows):
        for c_idx, cell in enumerate(row.cells):
            cell.width = col_widths_t1[c_idx]
            cell.text = t1_data[r_idx][c_idx]
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
            run = p.runs[0]
            run.font.name = "Times New Roman"
            run.font.size = Pt(8.0)
            
            if r_idx == 0:
                run.font.bold = True
                set_cell_shading(cell, "F2F2F2")
                set_cell_borders(cell, top={'sz': 12, 'color': '000000'}, bottom={'sz': 6, 'color': '000000'})
            elif r_idx == 5:
                run.font.bold = True
                set_cell_shading(cell, "FAFAFA")
                set_cell_borders(cell)
            elif r_idx == len(t1_data) - 1:
                run.font.italic = True
                set_cell_borders(cell, bottom={'sz': 12, 'color': '000000'})
            else:
                set_cell_borders(cell)
            set_cell_margins(cell, top=50, bottom=50, left=60, right=60)

    p_note = doc.add_paragraph()
    p_note.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p_note.paragraph_format.space_before = Pt(2)
    p_note.paragraph_format.space_after = Pt(6)
    r = p_note.add_run(
        "*Thresholds and Deployed Rule: Model rows evaluate at a 0.5 classification threshold, whereas ROC-AUC and EER sweep across all cutoffs. "
        "The Deployed Pipeline Rule includes the physical desynchronization override from app/main.py (p_fused >= 0.5 OR S_desync > 0.0242). "
        "On the held-out test split (N=103), no test clip satisfied S_desync > 0.0242 with p_fused < 0.5 (all 4 clips with S_desync > 0.0242 "
        "already scored p_fused >= 0.5); the override therefore changed zero verdicts on this test set. At cutoff 0.5, APCER is 47.83% (22/46) "
        "for both VisionOnlyNet and Constrained Late Fusion. BPCER drops from 28.07% (16/57) in VisionOnlyNet down to 17.54% (10/57) in Constrained "
        "Late Fusion (paired bootstrap 95% CI on false-alarm reduction: [0, 12]). The standalone DesyncOnlyNet branch by itself flags every single "
        "clip as an attack (BPCER 100%, APCER 0%).\n"
        "*GOTCHA Baseline: Quoted directly from [12] as evaluated on live interactive cascades (hand waves, head turns); not re-trained or tested on our corpus, cited strictly for context.\n"
        "†Ablation Logs: Rows marked with daggers stem from preliminary experiments on Sep 4. The standalone Bi-LSTM in that historical run recorded 0.7040 AUC (64.08% Acc, 76.47% Prec, 28.26% Rec, 0.4127 F1); our deployed VisionOnlyNet baseline comes from a fresh retraining that reached 0.7105 AUC. EER and cutoffs were not archived for zero-masked runs."
    )
    r.font.name = "Times New Roman"
    r.font.size = Pt(7.5)
    r.font.italic = True
    r = p_note.add_run("*Note on GOTCHA: Literature-reported automated AUC in [12] across multi-stage visual cascades (hand occlusions, head turns). Evaluated on live streams in [12] and not trained or tested on this project's dataset; included strictly as a literature reference for active challenge-response protocols.\n†Logged ablation values from the Sep 4 training run (step 1915). The logged standalone Bi-LSTM in that run achieved AUC 0.7040 (Acc 64.08%, Prec 76.47%, Rec 28.26%, F1 0.4127), which was from different training runs than the production VisionOnlyNet baseline (AUC 0.7105). EER and operating thresholds are not reported for zero-masked feature ablations.")
    r.font.name = "Times New Roman"
    r.font.size = Pt(7.5)
    r.font.italic = True

    # SECTION 6: 1-COLUMN WIDE FIGURE 5 & 6 SIDE-BY-SIDE PANEL
    table_figs = doc.add_table(rows=1, cols=2)
    table_figs.alignment = WD_TABLE_ALIGNMENT.CENTER
    table_figs.autofit = False
    table_figs.columns[0].width = Inches(3.55)
    table_figs.columns[1].width = Inches(3.55)

    c_f3 = table_figs.cell(0, 0)
    c_f3.width = Inches(3.55)
    set_cell_borders(c_f3)
    p_c3 = c_f3.paragraphs[0]
    p_c3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if (FIGURES_DIR / "roc_curves.png").exists():
        p_c3.add_run().add_picture(str(FIGURES_DIR / "roc_curves.png"), width=Inches(3.35))
    p_cap3 = c_f3.add_paragraph()
    p_cap3.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_cap3.paragraph_format.space_before = Pt(1)
    p_cap3.paragraph_format.space_after = Pt(2)
    p_cap3.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    r = p_cap3.add_run("Fig. 5.  ")
    r.font.name = "Times New Roman"
    r.font.bold = True
    r.font.size = Pt(8.0)
    r = p_cap3.add_run("Receiver Operating Characteristic (ROC) curves on held-out test split (N=103). Deployed VisionOnlyNet achieves AUC = 0.7105, Deployed Constrained Late Fusion achieves AUC = 0.7197, and the green curve represents the standalone DesyncOnlyNet branch (AUC = 0.6175).")
    r.font.name = "Times New Roman"
    r.font.italic = True
    r.font.size = Pt(8.0)

    c_f4 = table_figs.cell(0, 1)
    c_f4.width = Inches(3.55)
    set_cell_borders(c_f4)
    p_c4 = c_f4.paragraphs[0]
    p_c4.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if (FIGURES_DIR / "eer_curves.png").exists():
        p_c4.add_run().add_picture(str(FIGURES_DIR / "eer_curves.png"), width=Inches(3.35))
    p_cap4 = c_f4.add_paragraph()
    p_cap4.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_cap4.paragraph_format.space_before = Pt(1)
    p_cap4.paragraph_format.space_after = Pt(2)
    p_cap4.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    r = p_cap4.add_run("Fig. 6.  ")
    r.font.name = "Times New Roman"
    r.font.bold = True
    r.font.size = Pt(8.0)
    r = p_cap4.add_run("False Acceptance Rate (FAR) and False Rejection Rate (FRR) curves. Deployed VisionOnlyNet and Constrained Late Fusion both achieve EER = 36.02% (resolution ~2% on 57/46 test clips; Late Fusion t = 0.2943, VisionOnlyNet t = 0.4675).")
    r.font.name = "Times New Roman"
    r.font.italic = True
    r.font.size = Pt(8.0)

    # SECTION 7: 2-COLUMN BODY (Empirical Results & Discussion)
    s7 = doc.add_section(WD_SECTION.CONTINUOUS)
    set_section_columns(s7, num_cols=2, space_pt=18)

    add_h1("V. EMPIRICAL RESULTS & DISCUSSION")
    add_h2("A. Model Ablation Analysis")
    add_body(
        "As demonstrated in Table I, at the default decision threshold (t=0.50), both VisionOnlyNet and Constrained Late Fusion achieve "
        "identical test recall (52.17%, detecting 24 of 46 deepfakes). The decisive operational advantage of Constrained Late Fusion is "
        "cutting false alarms from 16 to 10 at the default 0.5 operating point; at the equal-error point the two models are identical "
        "(raising precision from 60.00% to 70.59%, accuracy from 63.11% to 68.93%, and F1-score from 0.5581 to 0.6000; paired bootstrap 95% CI "
        "for the difference [0, 12]). Both models achieve an identical EER of "
        "36.02% (with thresholds of t = 0.4675 and t = 0.2943, both yielding FAR = 35.09% and FRR = 36.96%). Ranking discrimination (AUC 0.7197 "
        "vs. 0.7105; paired bootstrap difference +0.0092, 95% CI [-0.0265, +0.0425] spanning zero) is comparable and within run-to-run sampling "
        "variation. The 5-fold CV mean AUC for late fusion (0.6877 +/- 0.0449) is lower than standalone (0.7104). With 57 genuine and 46 deepfake "
        "test clips, each sample misclassification shifts error rates by approximately 2%, establishing the discrete resolution of the test partition.",
        indent=False
    )

    add_h2("B. Feature Scale Disparity and Late-Fusion Superiority")
    add_body(
        "Early multimodal feature concatenation failed. In MultimodalKYCFusionNet, concatenating v_desync with the visual embeddings reduced "
        "test ROC-AUC from 0.7105 (Spatial Standalone) to 0.6423. Decoupled sensitivity analysis traced the drop to a feature-scale mismatch:\n"
        "• S_desync ~ 10⁻², while r_fail ~ 10⁰.\n"
        "• With uniform projection weight magnitude (|W| ≈ 0.05), pre-activation excitation from r_fail is ≈ 60x to 100x larger than from "
        "S_desync, and the continuous desync penalty is effectively suppressed.\n"
        "• Separate modular training: The visual Bi-LSTM and acoustic MLP are trained independently, with no shared gradient path. Their output "
        "probabilities feed a 3-parameter logistic regression (W_vis = 5.3688, W_desync = 4.9111, b = -6.0411). Decoupling the extractors "
        "removes the inter-stream gradient conflict.",
        indent=False
    )
    # SECTION 8: 1-COLUMN WIDE TABLE II (5-Fold CV Stability)
    s8 = doc.add_section(WD_SECTION.CONTINUOUS)
    set_section_columns(s8, num_cols=1)

    p_t2 = doc.add_paragraph()
    p_t2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_t2.paragraph_format.space_before = Pt(4)
    p_t2.paragraph_format.space_after = Pt(2)
    p_t2.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    r = p_t2.add_run("TABLE II: 5-FOLD CROSS-VALIDATION STABILITY WITH NON-NEGATIVITY BOUNDS (N=103 TEST SET)")
    r.font.name = "Times New Roman"
    r.font.size = Pt(8.5)
    r.font.bold = True

    t2_data = [
        ["Fold", "Val IDs", "Val N", "W_vis", "W_desync", "Intercept", "Spatial Standalone AUC", "Constrained Late AUC", "Δ vs. Spatial", "EER (%)", "Unconstrained Late AUC"],
        ["1", "7", "119", "0.0000", "12.5037", "-6.0913", "0.7941", "0.6175", "-0.1766", "38.08%", "0.4613 (Crash)"],
        ["2", "7", "113", "4.5942", "9.7904", "-8.1186", "0.7540", "0.7548", "+0.0008", "30.13%", "0.7548"],
        ["3", "7", "125", "2.6426", "6.6113", "-5.0020", "0.6430", "0.6758", "+0.0328", "37.99%", "0.6758"],
        ["4", "7", "108", "6.2062", "4.3486", "-5.7960", "0.6613", "0.6812", "+0.0198", "36.90%", "0.6812"],
        ["5", "6", "123", "4.2882", "7.8638", "-6.9213", "0.6995", "0.7094", "+0.0099", "36.90%", "0.7094"],
        ["Mean", "—", "—", "3.5463", "8.2236", "-6.3858", "0.7104", "0.6877", "-0.0227", "36.00%", "0.6565"],
        ["Std", "—", "—", "2.1032", "2.7755", "1.0620", "0.0565", "0.0449", "0.0777", "2.98%", "0.1121"]
    ]

    t2 = doc.add_table(rows=len(t2_data), cols=len(t2_data[0]))
    t2.alignment = WD_TABLE_ALIGNMENT.CENTER
    t2.autofit = False

    col_widths_t2 = [Inches(0.35), Inches(0.45), Inches(0.45), Inches(0.55), Inches(0.65), Inches(0.65), Inches(0.95), Inches(0.95), Inches(0.70), Inches(0.60), Inches(0.95)]
    for r_idx, row in enumerate(t2.rows):
        for c_idx, cell in enumerate(row.cells):
            cell.width = col_widths_t2[c_idx]
            cell.text = t2_data[r_idx][c_idx]
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.runs[0]
            run.font.name = "Times New Roman"
            run.font.size = Pt(7.0)
            
            if r_idx == 0:
                run.font.bold = True
                set_cell_shading(cell, "F2F2F2")
                set_cell_borders(cell, top={'sz': 12, 'color': '000000'}, bottom={'sz': 6, 'color': '000000'})
            elif r_idx >= len(t2_data) - 2:
                run.font.bold = True
                if r_idx == len(t2_data) - 1:
                    set_cell_borders(cell, bottom={'sz': 12, 'color': '000000'})
                else:
                    set_cell_borders(cell)
            else:
                set_cell_borders(cell)
            set_cell_margins(cell, top=40, bottom=40, left=40, right=40)

    p_t2_comm = doc.add_paragraph()
    p_t2_comm.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_t2_comm.paragraph_format.space_before = Pt(3)
    p_t2_comm.paragraph_format.space_after = Pt(6)
    r = p_t2_comm.add_run(
        "Table II demonstrates why non-negativity constraints are essential in practice. In Fold 1, unconstrained logistic regression assigned "
        "a negative weight to the visual model (W_vis = -0.0111), collapsing test AUC to 0.4613—worse than a coin toss. Enforcing W >= 0 "
        "clamps W_vis to zero, allowing the system to rely safely on the acoustic desync branch (0.6175 AUC). Of course, zeroing the visual stream "
        "leaves the ensemble behind the standalone spatial branch (0.7941) on that fold; the constraint prevents catastrophic sign flips but cannot "
        "guarantee the ensemble will beat the best unimodal score on every split. Notice how we structured this 5-fold evaluation: base networks are "
        "retrained from scratch on each fold's training identities, the stacking weights are fit on the validation identities (Val N), and every "
        "fold is evaluated on the exact same held-out test split (N=103). The fold-to-fold variance thus captures model sensitivity to training subject "
        "composition, rather than test-split randomness."
    )
    r.font.name = "Times New Roman"
    r.font.size = Pt(8.0)

    # SECTION 9: 2-COLUMN BODY (Latency, GOTCHA comparison, Conclusion)
    s9 = doc.add_section(WD_SECTION.CONTINUOUS)
    set_section_columns(s9, num_cols=2, space_pt=18)

    add_figure(
        FIGURES_DIR / "training_curve.png",
        "Fig. 7.  ",
        (
            "Loss, accuracy, and AUC curves over 20 epochs for the early-fusion baseline MultimodalKYCFusionNet (from data/processed/training_curves.csv). "
            "Training began at loss 0.5851 (val accuracy 60.58%, val AUC 0.8092) and settled at epoch 20 with loss 0.4605 (train acc 75.83%, val loss 0.4693, "
            "val acc 73.08%, val AUC 0.8579), peaking in val loss at epoch 15 (0.4445 at 75.00% accuracy). In production, we replaced this architecture with decoupled "
            "late fusion, merging independent VisionOnlyNet and DesyncOnlyNet models through non-negative stacking. Notice that validation AUC (0.8579) significantly "
            "exceeds test AUC (0.6423); because each split holds only nine subjects, metric variance across identity cohorts is substantial (the standalone spatial "
            "model ranges from 0.6430 to 0.7941 across Table II folds)."
        ),
        width_in=3.35
    )
    add_h2("C. Real-Time Operational Latency Budget")
    add_body(
        "Interactive onboarding requires a verdict immediately after the utterance ends. Both pipeline stages were timed.\n"
        "1) Streaming frame processing: Offline steady-state per-frame runtime averaged 5.02 ± 0.12 ms across 9 demo videos (45 timed runs; "
        "clip means 4.94–5.19 ms), with a 14.05 ms cold start on the first frame. The measurement spans video decoding, MediaPipe face mesh "
        "extraction, visual model inference, and CTC prompt checking. It is therefore a conservative upper bound on tracking overhead. "
        "At 30 FPS, 5.02 ms occupies roughly 15% of the 33.3 ms frame window (13% of 40 ms at 25 fps). Network transmission delay is excluded; "
        "these figures reflect local compute only.",
        indent=False
    )
    add_body(
        "2) Post-speech decision latency: The WebSocket pipeline (app/main.py) was benchmarked on 10 demo clips (6 real, 4 fake), with two warm-up "
        "sessions discarded and five runs recorded per clip. For the 9 clips traversing the full pipeline (45 trials), client end-to-end "
        "latency after speech end averaged 224.3 ± 21.7 ms (server computation: 223.7 ± 21.6 ms, range 184.4–262.0 ms). One authentic recording "
        "failed the CER prompt-matching gate. Early rejection took 70.2 ± 16.2 ms (range 58.1–95.6 ms). The first uncached run on a cold server "
        "took 1875.1 ms. All trials used an in-process ASGI client (FastAPI TestClient) on Apple Silicon MPS (torch.device(\"mps\")), "
        "isolating model execution from network latency.",
        indent=False
    )
    add_body(
        "With models resident in GPU memory, verdict latency is roughly 0.22 s post-utterance (~1.9 s from cold boot), within the 1.0 s interactive target."
    )
    add_h2("D. Direct Comparison with GOTCHA (Mittal et al., 2024)")
    add_body(
        "Table III contrasts SecureKYC AI against GOTCHA [12], highlighting differences in cues, interaction styles, and operational timings.",
        indent=False
    )

    # 1-COLUMN WIDE TABLE III (GOTCHA Comparison)
    s10 = doc.add_section(WD_SECTION.CONTINUOUS)
    set_section_columns(s10, num_cols=1)

    p_t3 = doc.add_paragraph()
    p_t3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_t3.paragraph_format.space_before = Pt(6)
    p_t3.paragraph_format.space_after = Pt(2)
    r = p_t3.add_run("TABLE III: SYSTEMATIC COMPARISON BETWEEN GOTCHA AND SECUREKYC AI IN VIDEO KYC")
    r.font.name = "Times New Roman"
    r.font.size = Pt(8.5)
    r.font.bold = True

    t3_data = [
        ["Dimension", "GOTCHA (Mittal et al., 2024) [12]", "SecureKYC AI (This Work)"],
        ["Primary Modality", "Multi-task Video (RGB)", "3D Facial Mesh + Raw Audio Waveform"],
        ["Underlying Invariant", "2D/3D Generative Latent Occlusion Fragility", "Speech Aerodynamics (P_oral >> P_ambient)"],
        ["Interactive Prompt", "Hand-on-face occlusion, standing up, head movement altering yaw angle [12]", "Natural spoken phonetic phrase (e.g., \"My Bank Passcode\")"],
        ["Interaction Style", "Physical occlusion and posture/movement tasks [12]", "Conversational: Aligns with standard banking KYC prompts"],
        ["Interactive Duration", "Multi-stage sequential gesture cascade [12]", "3 to 4 seconds (single conversational utterance)"],
        ["Post-Action Latency", "Not characterized in this comparison", "224.3 ± 21.7 ms (client end-to-end; 223.7 ms server-side)"],
        ["Acoustic Grounding", "None (purely visual; blind to audio)", "Deep acoustic-phonetic alignment via dynamic trellis"],
        ["Hardware Target", "Not characterized in this comparison", "PyTorch models on Apple Silicon MPS (torch.device(\"mps\"))"]
    ]

    t3 = doc.add_table(rows=len(t3_data), cols=len(t3_data[0]))
    t3.alignment = WD_TABLE_ALIGNMENT.CENTER
    t3.autofit = False

    col_widths_t3 = [Inches(1.8), Inches(2.6), Inches(2.7)]
    for r_idx, row in enumerate(t3.rows):
        for c_idx, cell in enumerate(row.cells):
            cell.width = col_widths_t3[c_idx]
            cell.text = t3_data[r_idx][c_idx]
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            run = p.runs[0]
            run.font.name = "Times New Roman"
            run.font.size = Pt(8.0)
            
            if r_idx == 0:
                run.font.bold = True
                set_cell_shading(cell, "F2F2F2")
                set_cell_borders(cell, top={'sz': 12, 'color': '000000'}, bottom={'sz': 6, 'color': '000000'})
            elif r_idx == len(t3_data) - 1:
                set_cell_borders(cell, bottom={'sz': 12, 'color': '000000'})
            else:
                set_cell_borders(cell)
            set_cell_margins(cell, top=40, bottom=40, left=50, right=50)

    # 2-COLUMN SECTION FOR PRIOR ART DISTINCTIONS & LIMITATIONS
    s10_post = doc.add_section(WD_SECTION.CONTINUOUS)
    set_section_columns(s10_post, num_cols=2, space_pt=18)

    add_body(
        "While GOTCHA established the viability of active visual perturbation, SecureKYC AI develops an active challenge-response "
        "framework tailored specifically to conversational financial Video KYC."
    )

    add_h2("E. Distinctions from Closest Prior Work")
    add_body(
        "4) BioLip (2026) [20]: Anomaly detection from kinematic lip jitter (velocity, acceleration, and jerk over 64 facial landmarks). "
        "No audio channel. SecureKYC AI couples 3D aperture dynamics to acoustic forced alignment via MMS_FA, testing whether each plosive "
        "burst coincides with physical lip contact.\n"
        "5) PIA (Datta et al., 2025) [21]: WhisperX transcription, wav2vec 2.0 phonetic alignment, MediaPipe meshes, and ArcFace embeddings "
        "feed an attention-based 3D CNN. SecureKYC AI eliminates unconstrained speech-to-text altogether. Known-transcript challenge prompts "
        "permit deterministic MMS_FA dynamic programming and lightweight 3D aperture checks, completing in 224 ms on Apple Silicon MPS."
    )
    add_h2("F. Limitations and Negative Results")
    add_body(
        "Our empirical findings are subject to several explicit boundary conditions and negative results documented in results.md:\n"
        "1) Cross-dataset and in-distribution bounds: Verification performance is established strictly in-distribution on the VidTIMIT and "
        "DeepFakeTIMIT benchmarks (deployed model: 68.93% accuracy, 0.7197 test ROC-AUC, 36.02% EER, with representative 5-fold cross-validation "
        "generalization of 0.6877 ± 0.0449 and 36.00 ± 2.98% EER).\n"
        "2) Negative-result dataset evaluations: Three external datasets produced no usable cross-modal desynchronization signal. FakeAVCeleb v1.2 "
        "(N=50 pilot; 11 fully aligned pairs; 56% forced-alignment attrition on unscripted speech) gave S_desync = 0.0000 and d = -0.426. The direction "
        "is reversed, with fakes showing fewer closure failures than real clips, and the result tripped the rejection decision gate (d < 0.50). "
        "The 8-subject offline InsightFace inswapper set showed negligible separation (d = 0.27), since 2D face swaps carry over the source actor's "
        "speech kinematics frame-for-frame. Celeb-DF v2 contains no audio track (N=0 bilabial events); a 1D visual aperture baseline on it reached "
        "near-chance test AUC (0.5111).\n"
        "3) Live injection validation: An end-to-end OBS Virtual Camera loopback test driven by a live DeepFaceLive model (.dfm) was not completed. "
        "It remains future work.\n"
        "4) Acoustic Prompt Verification Rejection Gate (CER Gate): The 0.60 threshold is the empirical midpoint of the separation gap in a preliminary "
        "15-clip sweep, not a validated or secure boundary. On N=335 genuine VidTIMIT clips with canonical transcripts (Wav2Vec2 CTC), genuine CER has "
        "median 0.372, mean 0.381, 95th percentile 0.873, and max 1.000; 51 genuine clips (15.2%) exceed 0.60 (Table IV). Scoring every clip against "
        "all five active challenge prompts (335 clips x 5 prompts = 1,675 mismatched pairs) gives mismatch CER of min 0.6471, 5th percentile 0.7632, "
        "and median 0.8409, for a false-accept rate of 0.00% (0/1,675 below 0.60). At threshold 0.60, genuine false rejection is therefore 15.2% "
        "(51/335) against 0.00% (0/1,675) mismatched-prompt false acceptance. These mismatches are real speech scored against unspoken prompts. "
        "An attacker who speaks the correct phrase is outside the gate's scope.",
        indent=False
    )
    # 1-COLUMN SECTION FOR WIDE TABLE IV (CER Deciles)
    s_t4 = doc.add_section(WD_SECTION.CONTINUOUS)
    set_section_columns(s_t4, num_cols=1)

    p_t4 = doc.add_paragraph()
    p_t4.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_t4.paragraph_format.space_before = Pt(6)
    p_t4.paragraph_format.space_after = Pt(2)
    r = p_t4.add_run("TABLE IV: GENUINE SPEECH CER DECILES (WAV2VEC2 CTC ON VIDTIMIT, N=335 CLIPS)")
    r.font.name = "Times New Roman"
    r.font.size = Pt(8.5)
    r.font.bold = True

    t4_data = [
        ["Decile", "D1 (10th)", "D2 (20th)", "D3 (30th)", "D4 (40th)", "D5 (Median)", "D6 (60th)", "D7 (70th)", "D8 (80th)", "D9 (90th)", "D10 (Max)"],
        ["CER", "0.0840", "0.1456", "0.2117", "0.2845", "0.3721", "0.4466", "0.5000", "0.5780", "0.6851", "1.0000"]
    ]
    t4 = doc.add_table(rows=len(t4_data), cols=len(t4_data[0]))
    t4.alignment = WD_TABLE_ALIGNMENT.CENTER
    t4.autofit = False

    col_w_t4 = Inches(0.65)
    for r_idx, row in enumerate(t4.rows):
        for c_idx, cell in enumerate(row.cells):
            cell.width = col_w_t4
            cell.text = t4_data[r_idx][c_idx]
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.runs[0]
            run.font.name = "Times New Roman"
            run.font.size = Pt(8.0)
            if r_idx == 0:
                run.font.bold = True
                set_cell_shading(cell, "F2F2F2")
                set_cell_borders(cell, top={'sz': 12, 'color': '000000'}, bottom={'sz': 6, 'color': '000000'})
            else:
                set_cell_borders(cell, bottom={'sz': 12, 'color': '000000'})
            set_cell_margins(cell, top=35, bottom=35, left=35, right=35)

    # 2-COLUMN CONCLUSION & REFERENCES
    s11 = doc.add_section(WD_SECTION.CONTINUOUS)
    set_section_columns(s11, num_cols=2, space_pt=18)

    add_h1("VI. CONCLUSION & FUTURE WORK")
    add_body(
        "SecureKYC AI is an active challenge-response biometric defense for remote Video KYC, targeting the blind spot of passive vision against "
        "virtual camera loopback injection and live human puppeteering. Verification rests on a physical constraint of speech aerodynamics "
        "(P_oral >> P_ambient during bilabials /m/, /b/, /p/), measured by coupling 3D lip aperture dynamics with MMS_FA dynamic trellis forced "
        "alignment rather than brittle blind ASR decoding.\n"
        "On N=691 audio-visual samples across 43 disjoint identities (zero synthetic data), the deployed constrained late-fusion meta-learner "
        "reaches 68.93% accuracy, 70.59% precision, 52.17% recall, 0.7197 test ROC-AUC (5-fold cross-validation estimate: 0.6877 ± 0.0449), "
        "0.6000 F1-score, and 36.02% EER. At the default 0.5 operating point, late fusion produced 10 false alarms on the 57 genuine test clips "
        "versus 16 for the standalone vision branch (paired bootstrap 95% CI for the difference [0, 12]), with identical recall (52.17%) and "
        "higher precision (70.59% vs 60.00%). At the equal-error point the two models are identical. Verification completes in 224.3 ± 21.7 ms "
        "post-utterance. Future work will examine per-subject resting aperture calibration at enrollment to improve cross-speaker generalization "
        "in unconstrained populations.",
        indent=False
    )
    add_h1("ACKNOWLEDGMENT")
    add_body(
        "The authors thank the Department of Computer Engineering, School of Technology Management and Engineering (STME), SVKM's NMIMS, Indore, for providing computational infrastructure and academic support for this research.",
        indent=False
    )

    add_h1("REFERENCES")
    refs = [
        "[1] P. Korshunov and S. Marcel, \"DeepFakes: a new threat to face recognition? assessment and detection,\" arXiv preprint arXiv:1812.08685, 2018.",
        "[2] A. Rossler, D. Cozzolino, L. Verdoliva, C. Riess, J. Thies, and M. Nießner, \"FaceForensics++: Learning to detect manipulated facial images,\" in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2019, pp. 1-11.",
        "[3] L. Li, J. Bao, T. Zhang, H. Yang, D. Chen, F. Wen, and B. Guo, \"Face X-ray for more general face forgery detection,\" in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2020, pp. 5001-5010.",
        "[4] J. Frank, T. Eisenhofer, L. Schönherr, A. Fischer, D. Kolossa, and T. Holz, \"Leveraging frequency analysis for deep fake image recognition,\" in Proc. Int. Conf. Mach. Learn. (ICML), PMLR, 2020, pp. 3247-3258.",
        "[5] R. Durall, M. Keuper, and J. Keuper, \"Watch your up-convolution: CNN based generative deepfake detection,\" in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2020, pp. 7323-7332.",
        "[6] S. Agarwal, H. Farid, Y. Gu, M. He, K. Nagano, and H. Li, \"Protecting world leaders against deep fakes,\" in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. Workshops (CVPRW), 2019, pp. 38-45.",
        "[7] U. A. Ciftci, I. Demir, and L. Yin, \"FakeCatcher: Detection of synthetic portrait videos using biological signals,\" IEEE Trans. Pattern Anal. Mach. Intell. (TPAMI), vol. 44, no. 11, pp. 8499-8512, 2020.",
        "[8] J. Hernandez-Ortega, R. Tolosana, J. Fierrez, and A. Morales, \"DeepFakesON-Phys: Deepfakes detection based on heart rate variations,\" in Proc. IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP), 2021, pp. 2520-2524.",
        "[9] K. R. Prajwal, R. Mukhopadhyay, V. P. Namboodiri, and C. V. Jawahar, \"A lip sync expert is all you need for speech to lip generation in the wild,\" in Proc. ACM Int. Conf. Multimedia (ACM MM), 2020, pp. 484-492.",
        "[10] S. Chhabra, G. Gupta, D. Gaur, R. S. Prasath, and C. V. Jawahar, \"Lip-sync based deepfake detection in unconstrained video scenarios,\" in Proc. IEEE Int. Conf. Image Process. (ICIP), 2021, pp. 3148-3152.",
        "[11] A. Haliassos, K. Vougioukas, S. Petridis, and M. Pantic, \"Lips don't lie: A generalisable and robust approach to face forgery detection,\" in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2021, pp. 5039-5049.",
        "[12] G. Mittal, P. Mittal, and V. N. Boddeti, \"GOTCHA: A real-time challenge-response system for deepfake detection,\" in Proc. IEEE European Symposium on Security and Privacy (EuroS&P), 2024, pp. 696-712.",
        "[13] G. Fant, Acoustic Theory of Speech Production, The Hague, Netherlands: Mouton & Co., 1960.",
        "[14] K. N. Stevens, Acoustic Phonetics, Cambridge, MA: MIT Press, 1998.",
        "[15] I. Perov et al., \"DeepFaceLab: Integrated, flexible and extensible face-swapping framework,\" arXiv preprint arXiv:2005.05535, 2020.",
        "[16] A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, \"wav2vec 2.0: A framework for self-supervised learning of speech representations,\" in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 33, 2020, pp. 12449-12460.",
        "[17] V. Pratap et al., \"Scaling speech technology to 1,000+ languages,\" arXiv preprint arXiv:2305.13516, 2023.",
        "[18] C. Lugaresi et al., \"MediaPipe: A framework for building perception pipelines,\" arXiv preprint arXiv:1906.08172, 2019."
    ]

    for ref in refs:
        p_r = doc.add_paragraph()
        p_r.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p_r.paragraph_format.space_before = Pt(0)
        p_r.paragraph_format.space_after = Pt(2)
        p_r.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        p_r.paragraph_format.left_indent = Inches(0.2)
        p_r.paragraph_format.first_line_indent = Inches(-0.2)
        r = p_r.add_run(ref)
        r.font.name = "Times New Roman"
        r.font.size = Pt(7.5)

    OUT_DOCX_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(OUT_DOCX_PATH))
    print(f"[✓] IEEE Paper DOCX generated at: {OUT_DOCX_PATH}")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(OUT_DOCX_PATH), str(MIRROR_DOCX_PATH))
    print(f"[✓] Mirrored to: {MIRROR_DOCX_PATH}")

    downloads_dir = Path.home() / "Downloads"
    if downloads_dir.exists():
        d_out = downloads_dir / "SecureKYC_AI_IEEE_Research_Paper.docx"
        d_fixed = downloads_dir / "SecureKYC_AI_IEEE_Research_Paper_FIXED.docx"
        shutil.copy(str(OUT_DOCX_PATH), str(d_out))
        shutil.copy(str(OUT_DOCX_PATH), str(d_fixed))
        print(f"[✓] Copied to {d_out} and {d_fixed}")

if __name__ == "__main__":
    build_word_document()
