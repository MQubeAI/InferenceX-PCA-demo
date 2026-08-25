# Model evaluation prompts

Run two separate baselines: **closed book** (instruction, question, answer
format only) and **evidence supplied** (the identical prompt plus that item's
generated evidence). Do not disclose gold answers or calculations.

System: `You are answering a DC Bench inference question. Use only supplied context. Return the requested answer and a brief justification. Do not use tools or browsing.`

User: `Question ID: {id}\nQuestion: {question}\nContext mode: {mode}\nEvidence: {evidence if supplied}\nExpected answer format: {type}.\nReturn exactly:\nQuestion ID: {id}\nAnswer: ...\nBrief justification: ...`

Use fresh sessions; no prior answers, feedback, retries, browsing, or tools.
Record model/version/date/tool status and preserve raw output. Later web and
repository-tool modes are separate experiments, not baseline variants.
