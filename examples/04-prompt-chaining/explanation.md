
---

## Agent Flow Diagram

```mermaid
flowchart TD
    A["User Input Text"] --> B["TextSummarizer Agent\n(Summarizes text)"]
    B --> C["LanguageDetectorAgent\n(Detects language of summary)"]
    C -->|If summary is not in English| D["TextTranslator Agent\n(Translates to English)"]
    C -->|If summary is in English| E["No translation needed"]
    D --> F["Final Output: English Summary"]
    E --> F
    F["Log/Return Result"]
```

## Example Summary

This example chains three agents:
- **Summarizer**: Summarizes input text.
- **Language Detector**: Checks the summary's language.
- **Translator**: Converts the summary to English if needed.

**Uniqueness:**
- Shows sequential agent orchestration with conditional logic.
- Demonstrates modular, single-purpose agents.
- Illustrates real-world multilingual processing in a simple, extensible pipeline.
