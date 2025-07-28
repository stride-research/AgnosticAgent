# Parallel Tool Use Example

This example demonstrates:

- **Parallel tool calls**: The agent invokes multiple tools simultaneously to answer a single prompt (e.g., fetching temperature and humidity together).
- **Efficiency**: Reduces latency by parallelizing independent tool calls.

**Comparison:**
- Unlike the sequential example, this approach is optimal when tool calls are independent and can be executed concurrently.
- Useful for scenarios where multiple data points are needed at once. 