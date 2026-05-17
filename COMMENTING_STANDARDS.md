# Senior Level Commenting Standards

These standards are based on the philosophy that code should be written for humans to read, prioritizing clarity and naming over excessive documentation.

## 1. Function and Method Level
- **Summary:** Write a sentence or two explaining what the function does.
- **Contract:** Clearly describe inputs and outputs.
- **Context:** Include a history of *why* the code exists if the rationale isn't immediately obvious from the implementation.
- **Style:** Stick to the facts. Comments should be concise and direct, not written like an English paper. Avoid verbosity; if comments are too long, people will skip or skim them, missing vital information.

## 2. In-line Comments
- **The Exception, Not the Rule:** Comments on single lines should be rare.
- **Usage:** Only comment on a single line if the logic is not immediately obvious. 
- **Goal:** Minimize these situations by refactoring complex logic into readable segments.

## 3. Naming Over Documentation
- **Priority:** Much more important than comments are names.
- **Function Names:** If you pick good names, you shouldn't need many comments. A name like `warn_if_book_is_overdue()` is self-documenting.
- **Variables:** Use descriptive names for local variables and loop iterators to make the flow obvious.

## 4. The "Time-Travel" Test
- The best test of your commenting and coding style is to revisit code you wrote several months ago.
- **Verification:** If you can't figure out what it does after a few months away, your colleagues will struggle even more. If the code isn't clear to you later, it needs better naming or more targeted comments.

## 5. Professional Context
- In a professional context, comments are the exception.
- While some stacks (like Java/Javadoc) may have specific conventions, the primary goal is always to make the code obvious by its name and structure.
