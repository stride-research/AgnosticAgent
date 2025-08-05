# Orchestrator-Worker Example

This example demonstrates:

- **Hierarchical agent orchestration**: An orchestrator agent manages and delegates tasks to multiple subagents (workers), each processing a section of input in parallel.
- **Recursive agent spawning**: Subagents are instances of the same class as the parent, enabling recursive, scalable workflows.
- **Parallelism at multiple levels**: Both the orchestrator and its subagents can run tasks in parallel, maximizing efficiency.
- **Complex document processing**: Designed for use cases like summarizing PDFs by section and extracting keywords per section.

**Comparison:**
- Goes beyond single-agent and flat tool orchestration by introducing multi-level, dynamic agent hierarchies.
- Demonstrates advanced workflow patterns (parallel, recursive, and hierarchical) not present in other examples.
- Useful for large-scale, modular processing tasks where work can be distributed and aggregated.