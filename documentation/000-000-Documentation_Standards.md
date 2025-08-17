# Documentation Standards for PTR Tool Filter

## File Naming Convention

All documentation files follow this pattern: `XXX-YYY-FileName.md`

Where:
- `XXX` = Category number (3 digits)
- `YYY` = Sequence number within category (3 digits)
- `FileName` = Descriptive name in PascalCase

## Category Numbers

### 000 - Meta Documentation
- `000-000-Documentation_Standards.md` (this file)

### 001 - Core Documentation
- `001-001-PTR_Paper_Summary.md` - Summary of the original PTR research paper
- `001-002-PTR_TRD.md` - Technical Requirements Document
- `001-003-PTR_TODO.md` - Implementation TODO list
- `001-004-MVP_Plan.md` - MVP implementation plan

### 001 - Technical Guides
- `001-005-Qdrant_Setup.md` - Qdrant vector database setup guide
- `001-006-evaluation_framework.md` - Evaluation framework using RAGAS/Phoenix/MLflow
- `001-007-litellm_embedding_guide.md` - LiteLLM embedding integration guide
- `001-008-vector_database_comparison.md` - Comparison of vector database options

### 002 - API Documentation (Future)
- `002-001-API_Reference.md`
- `002-002-API_Examples.md`

### 003 - Deployment Guides (Future)
- `003-001-Docker_Setup.md`
- `003-002-Kubernetes_Deployment.md`

### 004 - User Guides (Future)
- `004-001-Getting_Started.md`
- `004-002-Best_Practices.md`

## Document Structure

Each document should include:

1. **Title** - Clear, descriptive title matching the filename
2. **Overview** - Brief summary of the document's purpose
3. **Table of Contents** (for longer documents)
4. **Main Content** - Well-structured with proper headings
5. **References** - Links to related documents or external resources

## Formatting Guidelines

- Use proper Markdown heading hierarchy (# ## ### ####)
- Code blocks should specify the language for syntax highlighting
- Use tables for structured data comparisons
- Include diagrams where helpful (Mermaid or ASCII art)
- Keep line length reasonable for readability

## Cross-References

When referencing other documents, use relative links:
```markdown
See [MVP Plan](./001-004-MVP_Plan.md) for implementation details.
```

## Version Control

- Major updates should be noted in commit messages
- Consider adding a changelog section for living documents
- Archive old versions if major rewrites occur