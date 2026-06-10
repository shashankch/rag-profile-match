import sys
import time
from pathlib import Path

# Add src/ to path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import config
from resume_rag import ResumeRAGPipeline
from job_matcher import JobMatcher
from fs_tools import list_files, read_file
import numpy as np

print("Initializing pipeline...")
pipeline = ResumeRAGPipeline()

print("Ingesting directory...")
start_time = time.time()
pipeline.ingest_directory(config.RESUMES_DIR)
end_time = time.time()
print(f"Ingested all resumes in {end_time - start_time:.2f} seconds.")

matcher = JobMatcher()
jds = list_files(config.JOB_DESCRIPTIONS_DIR)

latencies = []

for jd_file in sorted(jds, key=lambda f: f['name']):
    jd_data = read_file(jd_file['path'])
    if not jd_data['success']:
        continue
    
    content = jd_data['content']
    print(f"\n{'='*60}")
    print(f"JD File: {jd_file['name']}")
    print(f"{'='*60}")
    print(content[:200] + "...\n")
    
    t0 = time.time()
    results = matcher.match(content, k=3)
    t1 = time.time()
    
    latency = (t1 - t0) * 1000
    latencies.append(latency)
    
    print(f"Retrieval latency: {latency:.2f} ms")
    print("Top Matches:")
    for idx, match in enumerate(results['top_matches']):
        print(f"  {idx+1}. {match['candidate_name']} (Score: {match['match_score']})")
        print(f"     Reasoning: {match['reasoning']}")
        print(f"     Excerpts: {match['relevant_excerpts'][0][:150]}...")
        print()

print("\nMatching Performance Metrics:")
print(f"- Average Match Latency: {np.mean(latencies):.2f} ms")
print(f"- Median Match Latency: {np.median(latencies):.2f} ms")
print(f"- Min Match Latency: {np.min(latencies):.2f} ms")
print(f"- Max Match Latency: {np.max(latencies):.2f} ms")
