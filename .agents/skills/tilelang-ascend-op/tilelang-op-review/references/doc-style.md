# Documentation Quality Review

<applicability>
Language: Markdown
Side: N/A
Domain: false
Enabled by default: true
</applicability>

<review_load>
General review subagent rule capacity limit: 4
</review_load>

## Purpose

Check Markdown documents for structural, expressive, and content-credibility issues.

## Quick Index

| Rule ID | Rule name | Severity |
|---------|-----------|----------|
| D1 | Complete Markdown structure | Medium |
| D2 | Consistent lists and terminology within a document | Medium |
| D3 | Clear formatting for commands, code, and identifiers | Medium |
| D4 | Valid paths, links, and factual references | Medium |

## Specialized Review Method

Report only issues supported by direct textual evidence or reproducible checks. Do not report preferences not specified by the repository, such as spacing between Chinese and English text or punctuation style. Every finding must include the document path, line number, observable evidence, and a remediation recommendation.

---

## D1: Complete Markdown Structure

**Issue description**: Unjustified heading-level jumps, unclosed code fences, mismatched table columns, or incorrect list nesting can alter rendered output and an agent's interpretation of section boundaries.

**Review method**: Check the heading hierarchy starting from the level-one heading. Verify each code fence's language and closing position. Render or parse complex tables, nested lists, and Mermaid diagrams. Report only errors that affect structure or meaning.

**Exclusion rules**: Ignore the number of blank lines, heading wording, and other purely personal formatting preferences unless the repository defines an explicit rule.

**Decision method**: Report an issue when the Markdown cannot be parsed correctly, content is assigned to the wrong section, or the displayed meaning changes.

## D2: Consistent Lists and Terminology Within a Document

**Issue description**: Mixing mutually exclusive forms within the same list level, or using different names for the same concept without explanation, makes requirement levels and subject identity ambiguous.

**Review method**: Check whether normative terms such as “must,” “should,” and “may,” as well as status names, API names, path names, and abbreviations, remain consistent across headings, body text, and tables. Confirm that any terminology change is explained in a glossary or by the surrounding context.

**Decision method**: Report an issue when the inconsistency allows two reasonable interpretations of the rule's subject, status, or requirement. Do not report punctuation-only differences that do not affect meaning.

## D3: Clear Formatting for Commands, Code, and Identifiers

**Issue description**: Unclear boundaries among commands, output, and pseudocode can cause users to copy incorrect content or cause agents to treat explanatory prose as an executable step.

**Review method**: Use language-tagged fences for multiline commands and code. Use inline code for filenames, APIs, and variable names. Clearly distinguish command placeholders from real arguments. Check that line continuations, quoting, working directories, and environment variables are sufficient for reproduction.

**Exclusion rules**: A single short command may use inline code. Illustrative code explicitly marked as non-executable does not need every piece of boilerplate.

**Decision method**: Report an issue when following the example as documented performs a different operation, causes a syntax error, or targets a dangerously broad scope.

## D4: Valid Paths, Links, and Factual References

**Issue description**: Broken paths and anchors block further investigation. Presenting version-dependent APIs, hardware constraints, or performance numbers as unsourced facts can make reviews and implementations rely on stale assumptions.

**Review method**: Resolve relative links from the document's directory and check the target files and heading anchors. Run key commands in the document, or at least verify that their entry points exist. Validate API facts against the currently imported source/lowering implementation, and check hardware and performance conclusions against the target SoC, measurement methodology, and raw results.

**Exclusion rules**: Content explicitly identified as a design candidate, illustrative value, or item requiring validation need not already have measured results, but it must avoid definitive wording such as “is supported” or “improves performance.”

**Decision method**: Report repository-local links or commands directly when they are demonstrably invalid. If external facts are inaccessible or their version is unknown, mark them for confirmation rather than declaring them true or false.
