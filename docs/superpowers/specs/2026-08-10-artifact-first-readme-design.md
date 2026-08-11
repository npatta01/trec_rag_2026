# Artifact-first repository landing design

## Goal

Make the public repository immediately understandable to a competition reviewer
or first-time visitor. The first action should be opening the published project
hub, while implementation and reproduction details remain available lower in
the README.

## Public identity

- Project description: **NP Labs submission for TREC RAG 2026.**
- Canonical public homepage:
  `https://npatta01.github.io/trec_rag_2026/`
- The GitHub repository description stays concise and uses the same project
  identity rather than enumerating individual runs.
- The GitHub repository homepage field points to the canonical public homepage.

## README hierarchy

1. Title: `NP Labs · TREC RAG 2026`.
2. One sentence identifying this as the NP Labs submission for TREC RAG 2026.
3. A prominent Markdown link labeled `View the project artifact hub` whose exact
   target is `https://npatta01.github.io/trec_rag_2026/`.
4. A concise overview explaining that the hub contains the final architecture,
   accepted submission records, Retrieval analysis, and RAG analyses.
5. A compact set of direct repository links for the submission ledger,
   architecture source, validation skill, and implementation reference.
6. A lower `Developer notes` section preserving cloning, submodule, testing, and
   agent-workflow material without competing with the primary artifact link.

The README will not use a badge wall, enumerate all five organizer files in the
opening, or duplicate the full artifact hub navigation.

## Verification

- Assert that the README's primary link points to the exact GitHub Pages hub.
- Assert that the repository description and homepage match the approved values.
- Check all local README links resolve to tracked files.
- Confirm the public hub URL returns HTTP 200 after publication.
