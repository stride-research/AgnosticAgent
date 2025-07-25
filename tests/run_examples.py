from concurrent.futures import ProcessPoolExecutor, as_completed
import os


examples = [
      "python3 -m examples.01-no-tools.run",
      "python3 -m examples.02-tools.hybrid.run", 
      "python3 -m examples.02-tools.parallel.run", 
      "python3 -m examples.02-tools.sequential.run", 
      #python3 -m examples.03-file-upload.run, 
      "python3 -m examples.04-prompt-chaining.run", 
      "python3 -m examples.05-orchestrator-worker.run" 
]
def run_tests():
      results = {}
      with ProcessPoolExecutor() as executor:
            futures = {}
            for example in examples:
                  future = executor.submit(os.system, example)    
                  futures[future] = example
            for future in as_completed(futures.keys()):
                  example = futures[future]
                  data = future.result()
                  results[example] = data
      print("="*40)
      print(f"RESULTS => {results}")
      print("="*40)

if __name__ == "__main__":
      run_tests()
            