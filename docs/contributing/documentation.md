# Documentation Guide

CircuitKit's documentation uses MkDocs with the Material theme, mkdocstrings for auto-generated API signatures, and Mermaid for architecture diagrams.

---

## Setup

```bash
pip install -e ".[docs]"
```

This installs: `mkdocs`, `mkdocs-material`, `pymdown-extensions`, `mkdocstrings[python]`, `mkdocs-jupyter`, `pygments`.

---

## Building Locally

```bash
# Serve with auto-reload
mkdocs serve

# Build static site
mkdocs build

# CI-standard strict build (fails on broken links and warnings)
mkdocs build --strict
```

Open `http://127.0.0.1:8000` in your browser for the live preview.

---

## File Structure

All docs source is in `docs/`. The `mkdocs.yml` at the project root controls navigation and plugins.

```text
docs/
├── index.md              # Landing page
├── assets/               # CSS, JS, SVG logo, images
└── {section}/            # One directory per nav section
```

---

## Writing New Pages

### Where to put it

New pages go in the appropriate section directory. Add the path to `nav:` in `mkdocs.yml`.

### Conventions

- **Page title**: `# Title` (H1 at the top)
- **Sections**: `## Section` (H2) and `### Subsection` (H3)
- **Tables**: GitHub-Flavored Markdown
- **Code blocks**: fenced with language tag
- **Admonitions**: `!!! note`, `!!! warning`, `!!! tip`
- **Cross-links**: relative Markdown links, e.g. `[Pipeline](../user-guide/pipeline-overview.md)`
- **End each page** with `## Next Steps` linking to 2-3 related pages

### Admonitions

```markdown
!!! note "Title (optional)"
    Content here.

!!! warning
    This is a warning.

!!! tip
    This is a tip.
```

### Mermaid Diagrams

Use fenced mermaid blocks:

````markdown
```mermaid
flowchart LR
    A[Discover] --> B[Evaluate]
    B --> C[Intervene]
```text
````

### Math (MathJax)

Use `$...$` for inline and `$$...$$` for block math:

```markdown
The patching score is:

$$P_1 = \frac{\text{circuit avg} - \text{random avg}}{\text{baseline avg} - \text{random avg}}$$
```

### Auto-generated API Signatures

Use mkdocstrings directives to pull signatures from source:

```markdown
::: circuitkit.quick.discover
    options:
      show_source: false
      heading_level: 4
```

---

## Updating the Nav

Add new pages to `mkdocs.yml` under `nav:`:

```yaml
nav:
  - User Guide:
      - My New Page: user-guide/my-new-page.md
```

---

## Style Guide

- **Direct professional tone** — no filler phrases ("In this section, we will...")
- **Working code examples** for every concept
- **Tables** for comparisons and parameter references
- **No inline HTML** (except for unavoidable cases)
- **No bare links** — always use `[Link Text](url.md)`

---

## CI Check

The CI pipeline runs `mkdocs build --strict`. A broken internal link or a missing nav entry will fail the build. Run it locally before submitting:

```bash
mkdocs build --strict
```

---

## Next Steps

- [Development Setup](setup.md) — getting the dev environment running
- [Code Standards](standards.md) — Python style guide
