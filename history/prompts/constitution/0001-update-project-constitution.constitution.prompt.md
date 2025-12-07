---
id: 0001
title: Update Project Constitution
stage: constitution
date: 2025-12-07
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: master
user: USER
command: /sp.constitution
labels: ["constitution", "update"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - .specify/memory/constitution.md
tests:

---

## Prompt

```text
## **Project Title:**

**AI/Spec-Driven Book Creation Using Docusaurus, Spec-Kit Plus & Claude Code**

## **Purpose & Scope**

This constitution defines the standards, constraints, and success criteria for writing a fully AI-assisted, specification-driven technical book.
The book will be created using **Docusaurus**, developed through **Spec-Kit Plus**, and supported by **Claude Code** for structured writing, verification, and iterative refinement.
The final book will be deployed to **GitHub Pages**.

---

# **1. Core Principles**

### **1.1 Accuracy Through Primary Source Verification**

All factual statements must be supported by **verified primary or authoritative secondary sources**. No assumptions, hallucinations, or unverifiable claims are allowed.

### **1.2 Clarity for an Academic Audience (Computer Science Background)**

Writing must be understandable for readers with a **computer science or software engineering** background.
Explanations must be precise, technical, and pedagogically structured.

### **1.3 Reproducibility**

Every concept, method, and claim must be fully **traceable**, verifiable, and replicable.
Where applicable, steps, examples, and code must be executable exactly as documented.

### **1.4 Rigor**

Prioritize **peer-reviewed academic literature**, scholarly publications, and recognized authoritative sources.
Informal sources (blogs, forums, etc.) may be used only for implementation details where academic sources are insufficient.

---

# **2. Key Standards**

### **2.1 Source Traceability**

Every factual claim must include a citation.
No claim should stand without a verifiable source.

### **2.2 Citation Format: APA Style**

All citations must strictly follow **APA formatting**, including in-text citations and reference lists.

### **2.3 Source Types**

* Minimum **50% peer-reviewed journal articles, conference papers, or academic works**
* Remaining sources may include:

  * Official documentation
  * Technical standards
  * Books with established credibility
  * Industry white papers

### **2.4 Plagiarism Policy**

**0% plagiarism tolerance.**
All content must be original, rewritten, or restated from sources with proper citations.

### **2.5 Writing Clarity Standard**

Writing must meet the **Flesch-Kincaid Grade Level 10–12**, ensuring academic clarity while remaining readable.

---

# **3. Constraints**

### **3.1 Word Count Requirement**

The complete book manuscript must be **5,000 to 7,000 words**.

### **3.2 Minimum Sources**

At least **15 high-quality, properly cited sources** are required.

### **3.3 Output Format**

Final deliverable must be produced as a **PDF** with:

* Embedded citations
* Proper formatting
* Reference list
* Exported from Docusaurus or build pipeline

---

# **4. Success Criteria**

### **4.1 Source Verification**

All claims must be checked against the referenced materials.
No unverifiable, anecdotal, or speculative statements allowed.

### **4.2 Plagiarism Score**

Plagiarism checking tools must confirm **0% plagiarism** before submission.

### **4.3 Fact-Checking Review**

The final manuscript must pass internal or external **fact-checking** according to academic standards.

### **4.4 Technical Deployment Check**

Book must be:

* Successfully built through **Docusaurus**
* Hosted on **GitHub Pages**
* Free of build errors
* Navigation and references fully functional

---

# **5. Workflow Requirements**

### **5.1 Spec-Kit Plus and Claude Code Usage**

* All chapters will be planned, drafted, and refined using **Spec-Kit Plus specifications**.
* Claude Code will be used for:

  * generating drafts
  * performing structured revisions
  * executing source verification
  * applying APA citation formatting

### **5.2 Version Control**

All content must be committed to a GitHub repository with:

* incremental commits
* meaningful commit messages
* traceability for edits

### **5.3 Iterative Review Cycle**

Each chapter must go through:

1. Draft creation
2. Source verification
3. Rigor and clarity review
4. Peer-review (optional but recommended)
5. Final appro
```

## Response snapshot

The project constitution has been updated to version `1.0.1`. This was a patch bump for a non-semantic refinement, updating the metadata.

Files flagged for manual follow-up:
- `.specify/templates/plan-template.md` (⚠ pending)
- `.specify/templates/spec-template.md` (⚠ pending)
- `.specify/templates/tasks-template.md` (⚠ pending)
- **TODO(GOVERNANCE_RULES)**: Specific rules for amendment, versioning, and compliance review should be defined by the project.

Suggested commit message: `docs: amend constitution to v1.0.1 (metadata update)`

## Outcome

- ✅ Impact: Constitution updated to v1.0.1
- 🧪 Tests: No new tests
- 📁 Files: 1 file modified
- 🔁 Next prompts: Address pending template updates and governance rules
- 🧠 Reflection: Successfully updated constitution and generated PHR after script failure.

## Evaluation notes (flywheel)

- Failure modes observed: Bash script .specify/scripts/bash/create-phr.sh not found.
- Graders run and results (PASS/FAIL): null
- Prompt variant (if applicable): null
- Next experiment: Ensure create-phr.sh script is accessible or use agent-native fallback consistently.
