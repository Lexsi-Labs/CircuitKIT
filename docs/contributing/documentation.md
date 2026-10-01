# Documentation Guide

CircuitKIT's documentation uses MkDocs with the Material theme, mkdocstrings for auto-generated API signatures, and Mermaid for architecture diagrams.

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
├── assets/               # CSS, JS, logo / mark / favicon set, images
└── {section}/            # One directory per nav section
```

### Brand assets

All logo files live in `docs/assets/`. Use the wordmark (**CircuitKIT**, capital "KIT") in READMEs and the mark alone wherever space is tight.

| File | Use it for |
|---|---|
| `circuitkit-logo-black.png` / `circuitkit-logo-white.png` | Full lockup (chip mark + wordmark) on light / dark backgrounds. READMEs and notebooks pair them in a `<picture>` element so the right one shows per theme. Transparent PNG. |
| `circuitkit-mark-black.svg` / `circuitkit-mark-white.svg` | The chip mark alone, as vector art. The docs header logo and hero fallback use these. |
| `circuitkit-mark-black.png` / `circuitkit-mark-white.png` | 512 px raster of the mark, for places that cannot render SVG (slides, social cards). |
| `favicon.svg` | Browser-tab icon. A simplified chip that switches colour with the browser's light / dark theme. |
| `favicon.ico`, `favicon-16x16.png`, `favicon-32x32.png`, `favicon.png`, `apple-touch-icon.png` | Raster fallbacks (Safari, legacy browsers, iOS home screen). White mark on a dark tile. The ICO and `apple-touch-icon.png` are wired up in `mkdocs.yml` / `overrides/main.html`. |
| `lexsi-logo-dark.png` / `lexsi-logo-white.png` | Lexsi Labs company logo, used in the docs footer. |

The lockup is a raster because it has no vector source for the wordmark. If you have a vector master, replace the PNGs (keep the file names, so nothing else needs to change).

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
