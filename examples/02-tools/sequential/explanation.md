# Sequential Tool Use Example

This example demonstrates:

- **Stepwise tool orchestration**: The agent performs a series of tool calls in a strict order (e.g., add, then multiply).
- **Deterministic logic**: Each step depends on the result of the previous one.

**Comparison:**
- Contrasts with the parallel example, where tool calls are independent.
- Useful for workflows where each operation builds on the last, requiring strict sequencing. 