"""Prompt text. The JSON shape is enforced by structured outputs; prompts only explain meaning."""

_FIELD_RULES = """Rules for every field:
- value: copy it exactly as written in the document (keep currency symbols and date format). Use null when the document does not contain it. Never guess a value that is not printed.
- page: the 1-based page number where you found it, or null if the value is null.
- confidence: "high" if it is clearly printed, "medium" if you inferred it or it is partly legible, "low" if it is a guess."""

EXTRACT_PROMPT: dict[str, str] = {
    "invoice": f"""Extract the fields of this invoice.

vendor is who issued the invoice, buyer is who must pay. tax is the total tax amount (null if none is shown). total is the final amount due. line_items lists every billed line in order.

{_FIELD_RULES}""",
    "contract": f"""Extract the key terms of this contract.

parties lists every party that signs the contract. term is how long the contract lasts. amount is the total contract value or fee. termination_notice is how much notice is needed to end the contract.

{_FIELD_RULES}""",
}

ASK_PROMPT = """Answer the question using only the documents above. Each document is introduced by a line "Document N: name".

- answer: a short, direct answer. If the documents do not contain the answer, say so plainly.
- quotes: 1 to 3 quotes that support the answer. Each quote must be copied word for word from the document (no paraphrasing, at most 25 words). document is the 1-based document number, page is the 1-based page number. Use an empty list if nothing in the documents supports an answer.

Question: """
