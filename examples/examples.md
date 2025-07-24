Run with 
```python
      python3 -m examples.LLM.02-tools.run   
```  

FINAL EXAMPLE IDEA:
- Agents: orchestrator (plans, spawns specialized subagents), subagents(operate in parallel), memory summarizer (serving orchestaror)
- Goal: 
- Notes: by spwan it is meant that it can call any of the methods of the subagents. 

- HOW TO DO PARALLELISM (an LLM Agent calling other LLM Agent):
      - Subagents have well define toolkit + objective. Parent LLM give a single instruction. 
      - Get LLM to tell u all parallel procedures (LLM Agents + Prompt)
      - Execute parallel proecure as with coroutines/multiple processes