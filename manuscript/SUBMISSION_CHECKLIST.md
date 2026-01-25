# Pre-Submission Checklist for Expert Systems with Applications
## SAFE-Gate Manuscript Submission

**Journal:** Expert Systems with Applications (Elsevier)
**Manuscript Title:** SAFE-Gate: A Knowledge-Based Expert System for Emergency Triage Safety with Conservative Multi-Gate Architecture and Explainable Reasoning
**Date:** January 25, 2026
**Status:** Ready for Submission ✅

---

## 📋 JOURNAL REQUIREMENTS CHECKLIST

### 1. MANUSCRIPT FORMAT

#### ✅ Document Structure
- [x] **Template:** elsarticle.cls (Elsevier LaTeX template)
- [x] **Format:** Review mode (12pt, double-spaced, line numbers)
- [x] **Length:** 57 pages (within typical limit)
- [x] **File:** main.pdf (1.3 MB)

#### ✅ Front Matter
- [x] **Title:** Clear, descriptive, <150 characters
- [x] **Authors:** 2 authors with full names
- [x] **Affiliations:** Department of CS & IT, Naresuan University
- [x] **Corresponding author:** Chakkrit Snae Namahoot (chakkrits@nu.ac.th)
- [x] **ORCIDs:** Both authors have ORCIDs listed
- [x] **Email addresses:** Both authors

#### ✅ Abstract & Keywords
- [x] **Abstract structure:** Background, Objective, Methods, Results, Conclusions, Clinical Impact
- [x] **Abstract length:** ~500 words (within 200-250 word guideline - **⚠️ May need shortening**)
- [x] **Keywords:** 10 keywords provided (max 6 recommended - **⚠️ Reduce to 6**)

---

### 2. FIGURES & TABLES

#### ✅ Figure Quality
- [x] **Total figures:** 7 figures
- [x] **Format:** PDF (vector) for charts, PNG (high-res) for complex graphics
- [x] **Resolution:** 300 DPI or vector
- [x] **Size:** All figures readable at print size
- [x] **Color:** Professional, colorblind-friendly palette

#### ✅ Figure List
1. ✅ **Figure 1 (architecture.pdf):** System architecture - 269 KB, vector
2. ✅ **Figure 2 (safety_performance.pdf):** Safety performance - 39 KB, vector
3. ✅ **Figure 3 (per_class_metrics.pdf):** Per-class metrics - 25 KB, vector
4. ✅ **Figure 4 (confusion_matrix.png):** Confusion matrix - 178 KB, 300 DPI
5. ✅ **Figure 5 (baseline_comparison.pdf):** Baseline comparison - 37 KB, vector
6. ✅ **Figure 6 (risk_distribution.png):** Risk distribution - 90 KB, 300 DPI
7. ✅ **Figure 7 (support_distribution.png):** Support distribution - 133 KB, 300 DPI

#### ✅ Tables
- [x] **Total tables:** 3 tables (G1 rules, Overall performance, Per-class performance, Baseline comparison)
- [x] **Format:** LaTeX booktabs (professional)
- [x] **Captions:** All tables have descriptive captions

---

### 3. REFERENCES

#### ✅ Bibliography
- [x] **Style:** elsarticle-num (numbered citations)
- [x] **Format:** BibTeX (.bib file)
- [x] **Citations:** All references cited in text
- [x] **Quality:** Mix of recent (2020+) and seminal works
- [x] **Completeness:** Journal, conference, books included

#### ⚠️ To Review
- [ ] **Check all DOIs** are working
- [ ] **Verify author names** in references
- [ ] **Confirm publication years** are correct
- [ ] **Remove preprints** if published versions available

---

### 4. CONTENT REQUIREMENTS

#### ✅ Structure
- [x] **Introduction:** Clear problem statement, motivation, contributions
- [x] **Related Work:** Comprehensive literature review
- [x] **Methods:** Detailed architecture, algorithms, data generation
- [x] **Results:** Performance metrics, comparisons, ablations
- [x] **Discussion:** Findings, limitations, implications
- [x] **Conclusion:** Summary, contributions, future work

#### ✅ Technical Content
- [x] **Novelty:** Conservative merging architecture (not ensemble averaging)
- [x] **Validation:** 804 test cases, 6,400 synthetic data
- [x] **Comparison:** 4 baseline methods
- [x] **Metrics:** Sensitivity, specificity, precision, recall, F1
- [x] **Safety properties:** 6 formal properties with proofs

#### ✅ Reproducibility
- [x] **Algorithm:** Algorithm 1 (Conservative Merging) provided
- [x] **Equations:** All methods formalized mathematically
- [x] **Code availability:** GitHub repository mentioned
- [x] **Data statement:** Synthetic data generation described

---

### 5. ETHICAL COMPLIANCE

#### ✅ Ethics Statement
- [x] **IRB:** Not required (synthetic data only)
- [x] **Data privacy:** No patient data used directly
- [x] **Template data:** 320 real cases (IRB approved, used only as templates)
- [x] **Disclosure:** Clearly stated computational validation only

#### ✅ Conflict of Interest
- [ ] **Statement needed:** "The authors declare no competing financial interests or personal relationships that could have appeared to influence the work reported in this paper."

---

### 6. SUPPLEMENTARY MATERIALS

#### ✅ Code & Data
- [x] **Repository:** https://github.com/ChatchaiTritham/SAFE-Gate
- [x] **License:** MIT (mentioned in paper)
- [x] **Availability:** Public, open-source

#### ⚠️ To Prepare
- [ ] **README:** Ensure GitHub README is complete
- [ ] **Documentation:** API docs, usage examples
- [ ] **Reproducibility:** Scripts to regenerate figures
- [ ] **Requirements.txt:** Python dependencies listed

---

### 7. JOURNAL-SPECIFIC REQUIREMENTS

#### Expert Systems with Applications (ESA)

##### ✅ Scope Alignment
- [x] **Topic:** Expert systems, knowledge-based systems ✓
- [x] **Application:** Clinical decision support ✓
- [x] **Innovation:** Conservative knowledge integration ✓
- [x] **Impact:** Safety-critical medical AI ✓

##### ✅ Article Type
- [x] **Type:** Full-length research article
- [x] **Length:** ~15,000 words (within guideline)
- [x] **Figures:** 7 figures (reasonable for ESA)

##### ⚠️ Submission Requirements
- [ ] **Highlights:** 3-5 bullet points (85 characters max each) - **NEED TO CREATE**
- [ ] **Graphical Abstract:** Optional but recommended - **CONSIDER CREATING**
- [ ] **Cover Letter:** Required - **NEED TO CREATE**

---

## 🚨 CRITICAL ISSUES TO FIX

### HIGH PRIORITY

1. ⚠️ **Abstract Length**
   - Current: ~500 words
   - Guideline: 200-250 words
   - **Action:** Condense to 250 words maximum

2. ⚠️ **Keywords**
   - Current: 10 keywords
   - Guideline: Maximum 6
   - **Action:** Select 6 most relevant keywords

3. ⚠️ **Highlights**
   - Current: Not created
   - Required: 3-5 bullet points (85 chars max)
   - **Action:** Create highlights for submission system

4. ⚠️ **Cover Letter**
   - Current: Not created
   - Required: For submission
   - **Action:** Draft cover letter to Editor-in-Chief

### MEDIUM PRIORITY

5. ⚠️ **Conflict of Interest Statement**
   - Current: Not included
   - Required: Yes
   - **Action:** Add COI statement before References

6. ⚠️ **Data Availability Statement**
   - Current: In acknowledgments section
   - Better: Dedicated section before references
   - **Action:** Formalize data availability statement

7. ⚠️ **Author Contributions**
   - Current: Not specified
   - Optional: But recommended (CRediT taxonomy)
   - **Action:** Add author contribution statement

### LOW PRIORITY

8. ℹ️ **Graphical Abstract**
   - Current: Not created
   - Optional: But enhances visibility
   - **Action:** Consider creating visual summary

9. ℹ️ **Supplementary Material**
   - Current: Code on GitHub
   - Optional: Additional appendices
   - **Action:** Consider supplementary algorithms/proofs

---

## 📝 REQUIRED DOCUMENTS FOR SUBMISSION

### Mandatory Files
1. ✅ **main.pdf** - Manuscript (1.3 MB) - READY
2. ✅ **main.tex** - LaTeX source - READY
3. ✅ **references.bib** - Bibliography - READY
4. ✅ **figures/** - All 7 figures - READY
5. ⚠️ **highlights.txt** - 3-5 bullet points - **NEED TO CREATE**
6. ⚠️ **cover_letter.pdf** - Cover letter - **NEED TO CREATE**

### Optional But Recommended
7. ⚠️ **graphical_abstract.pdf** - Visual summary - **CONSIDER**
8. ⚠️ **supplementary.pdf** - Additional materials - **OPTIONAL**

---

## ✅ FINAL VERIFICATION CHECKLIST

### Before Submission
- [ ] **Re-read entire manuscript** for typos, grammar
- [ ] **Verify all cross-references** (figures, tables, equations, citations)
- [ ] **Check figure quality** when printed/zoomed
- [ ] **Compile LaTeX** one final time (no errors/warnings)
- [ ] **Test all hyperlinks** in PDF
- [ ] **Verify author info** (names, emails, affiliations, ORCIDs)
- [ ] **Check references** (complete, formatted correctly)
- [ ] **Prepare highlights** (3-5 bullets, 85 chars max)
- [ ] **Draft cover letter** (1 page)
- [ ] **Add COI statement**
- [ ] **Review submission guidelines** one more time

---

## 📊 MANUSCRIPT STATISTICS

| Metric | Value | Guideline | Status |
|--------|-------|-----------|--------|
| **Pages** | 57 | Typical: 40-60 | ✅ OK |
| **Word Count** | ~15,000 | Typical: 8,000-20,000 | ✅ OK |
| **Abstract** | ~500 words | Max: 250 | ⚠️ TOO LONG |
| **Keywords** | 10 | Max: 6 | ⚠️ TOO MANY |
| **Figures** | 7 | Typical: 6-10 | ✅ OK |
| **Tables** | 3 | Typical: 2-5 | ✅ OK |
| **References** | ~60 | Typical: 40-80 | ✅ OK |

---

## 🎯 NEXT STEPS (Priority Order)

### Step 1: Fix Critical Issues (TODAY)
1. **Condense Abstract** to 250 words
2. **Reduce Keywords** to 6
3. **Create Highlights** (3-5 bullets)
4. **Draft Cover Letter**
5. **Add COI Statement**

### Step 2: Final Review (TOMORROW)
6. Re-read manuscript carefully
7. Verify all references
8. Check all figures/tables
9. Test PDF compilation

### Step 3: Submit (DAY AFTER)
10. Upload to Elsevier Editorial System
11. Complete submission forms
12. Submit and await confirmation

---

## 📞 JOURNAL CONTACT

**Expert Systems with Applications**
- **Publisher:** Elsevier
- **ISSN:** 0957-4174
- **Impact Factor:** 8.5 (2023)
- **Quartile:** Q1 in Artificial Intelligence
- **Submission:** https://www.editorialmanager.com/eswa/
- **Guide for Authors:** https://www.elsevier.com/journals/expert-systems-with-applications/0957-4174/guide-for-authors

---

**Last Updated:** January 25, 2026
**Prepared by:** Claude Code Assistant
**Status:** 🟡 Near-ready (4 critical items to complete)
