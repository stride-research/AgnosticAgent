import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from .config import inline_args

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
    if inline_args.backend:
        # Run specific backend
        config = {"backend": inline_args.backend, "model": inline_args.model}
        if inline_args.backend == "ollama" and not inline_args.model:
            config["model"] = "qwen3:8b"
        elif inline_args.backend == "openrouter" and not inline_args.model:
            config["model"] = "google/gemini-2.5-pro"
        elif inline_args.backend == "openai" and not inline_args.model:
            config["model"] = "gpt-4o-mini"
        
        TEST_CONFIGURATIONS = [config]
    
    run_tests()
            