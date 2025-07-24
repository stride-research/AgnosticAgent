

- Make sure you can have the orchestrator recalling subagents  
- I want an orchestrator to:
      - run subagents in paralell
      - subagents are from the same class than the parent. 
      - subagent also have tasks in parallel


# Application context
- Objective: summarize pdf per sections, provide keywords per section 
- Architecture
  - ORM: Retrieves all text from pdf
  - Orchestrator: receives text, chunks into sections
  - Subagents (in parallel): summarizes a given section. Identifies keyword for a given chunk 
    - NOTE: This is an artificial example. Given the fact that the two aforementioned activities \
    solely depend on the LLM's internal knowledge (no external usage) you could trade some latency 
    by asking to do everything in a single prompt but saving significant prompts (no need to have a \
    duplicated input)