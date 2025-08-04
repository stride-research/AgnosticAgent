from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import argparse

# Test configurations for different backends
TEST_CONFIGURATIONS = [
    {"backend": "ollama", "model": "qwen3:8b"},
    {"backend": "openrouter", "model": "google/gemini-2.5-pro"},
]

# Examples to test - will be parameterized with backend and model
EXAMPLES = [
    "examples.01-no-tools.run",
    "examples.02-tools.hybrid.run", 
    "examples.02-tools.parallel.run", 
    "examples.02-tools.sequential.run", 
    "examples.04-prompt-chaining.run", 
    "examples.05-orchestrator-worker.run" 
]

def run_single_example(example_module, backend, model):
    """Run a single example with specified backend and model."""
    cmd = f"python3 -m {example_module} --backend {backend} --model {model}"
    return os.system(cmd)

def run_tests():
    """Run all examples with all test configurations."""
    results = {}
    
    # Build all test combinations
    test_cases = []
    for config in TEST_CONFIGURATIONS:
        backend = config["backend"]
        model = config["model"]
        for example in EXAMPLES:
            test_cases.append((example, backend, model))
    
    print(f"Running {len(test_cases)} test cases...")
    
    with ProcessPoolExecutor() as executor:
        futures = {}
        for example, backend, model in test_cases:
            future = executor.submit(run_single_example, example, backend, model)
            test_key = f"{example}|{backend}|{model}"
            futures[future] = test_key
        
        for future in as_completed(futures.keys()):
            test_key = futures[future]
            data = future.result()
            results[test_key] = data
    
    print("="*60)
    print("TEST RESULTS:")
    for test_case, result in results.items():
        status = "✓ PASS" if result == 0 else "✗ FAIL"
        print(f"{status} {test_case}")
    
    failed_tests = [k for k, v in results.items() if v != 0]
    if failed_tests:
        print(f"\nFailed tests ({len(failed_tests)}):")
        for test in failed_tests:
            print(f"  - {test}")
    
    print("="*60)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run examples with different LLM backends")
    parser.add_argument("--backend", choices=["ollama", "openrouter", "openai"], 
                       help="Specific backend to test (default: all)")
    parser.add_argument("--model", type=str, 
                       help="Specific model to test (default: backend defaults)")
    
    args = parser.parse_args()
    
    if args.backend:
        # Run specific backend
        config = {"backend": args.backend, "model": args.model}
        if args.backend == "ollama" and not args.model:
            config["model"] = "qwen3:8b"
        elif args.backend == "openrouter" and not args.model:
            config["model"] = "google/gemini-2.5-pro"
        elif args.backend == "openai" and not args.model:
            config["model"] = "gpt-4o-mini"
        
        TEST_CONFIGURATIONS = [config]
    
    run_tests()
            