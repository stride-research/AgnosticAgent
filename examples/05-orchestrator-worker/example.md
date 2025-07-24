

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
  - Subagents (in parallel): summarizes a given section. Identifies keyword for a given