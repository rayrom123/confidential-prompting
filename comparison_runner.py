"""
Comparison Runner: Test SPD with and without Freivalds integration
This script runs both versions and compares performance
"""

import subprocess
import time
import sys
import os

def run_experiment(version_name, script_name, master_cmd, worker_cmd, timeout=120):
    """Run a single experiment and capture results"""
    print(f"\n{'='*60}")
    print(f"🚀 RUNNING: {version_name}")
    print(f"{'='*60}")

    results = {
        'version': version_name,
        'master_output': '',
        'worker_output': '',
        'master_time': 0,
        'worker_time': 0,
        'success': False
    }

    try:
        # Start master process
        print(f"📊 Starting Master: {master_cmd}")
        master_process = subprocess.Popen(
            master_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.getcwd()
        )

        time.sleep(2)  # Wait for master to initialize

        # Start worker process
        print(f"🔧 Starting Worker: {worker_cmd}")
        worker_process = subprocess.Popen(
            worker_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.getcwd()
        )

        # Wait for completion or timeout
        start_time = time.time()

        try:
            master_output, master_errors = master_process.communicate(timeout=timeout)
            worker_output, worker_errors = worker_process.communicate(timeout=10)

            end_time = time.time()

            results['master_output'] = master_output
            results['worker_output'] = worker_output
            results['master_time'] = end_time - start_time
            results['success'] = True

            print("✅ Experiment completed successfully!")
            print(f"⏱️  Master time: {results['master_time']:.2f}s")
            print(f"⏱️  Worker time: {results['worker_time']:.2f}s")

        except subprocess.TimeoutExpired:
            print(f"⏰ Timeout after {timeout}s")
            master_process.terminate()
            worker_process.terminate()
            results['success'] = False

    except Exception as e:
        print(f"❌ Error running {version_name}: {e}")
        results['success'] = False

    return results

def analyze_results(results_without, results_with):
    """Analyze and compare the two results"""

    print(f"\n{'='*80}")
    print("📊 COMPARISON ANALYSIS")
    print(f"{'='*80}")

    # Success rate comparison
    print("\n🎯 SUCCESS COMPARISON:")
    print(f"  Without Freivalds: {'✅ SUCCESS' if results_without['success'] else '❌ FAILED'}")
    print(f"  With Freivalds:    {'✅ SUCCESS' if results_with['success'] else '❌ FAILED'}")

    # Performance comparison
    if results_without['success'] and results_with['success']:
        print("\n⚡ PERFORMANCE COMPARISON:")
        print(f"  Without Freivalds: {results_without['master_time']:.2f}s")
        print(f"  With Freivalds:    {results_with['master_time']:.2f}s")
        overhead = results_with['master_time'] - results_without['master_time']
        print(f"  Overhead: {overhead:+.2f}s ({overhead/results_without['master_time']*100:+.1f}%)")

    # Output comparison
    print("\n📝 OUTPUT COMPARISON:")
    master_out_without = results_without['master_output'][-500:] if results_without['success'] else "N/A"
    master_out_with = results_with['master_output'][-500:] if results_with['success'] else "N/A"

    print("  Without Freivalds output (last 500 chars):")
    print(f"    {master_out_without}")

    print("  With Freivalds output (last 500 chars):")
    print(f"    {master_out_with}")

    # Security analysis
    print("\n🔒 SECURITY ANALYSIS:")
    if results_without['success'] and results_with['success']:
        if "WARNING" in results_with['worker_output']:
            print("  ⚠️  Freivalds detected potential integrity issues")
        else:
            print("  ✅ Freivalds verified tensor integrity")

    print(f"\n{'='*80}")

def main():
    """Main comparison function"""

    # Commands for running experiments
    master_cmd = "python3 smd.py --standalone_master --model meta-llama/Llama-3.2-3B-Instruct --device cuda:0 --states_dir ./states --num_users 1 --timeout_sec 15 --max_num_tokens 2048 --print_idx 0"

    worker_cmd_without = "python3 smd_without_freivalds.py --standalone_worker --model meta-llama/Llama-3.2-3B-Instruct --device cuda:0 --states_dir ./states --user_id 0 --timeout_sec 15"

    worker_cmd_with = "python3 smd.py --standalone_worker --model meta-llama/Llama-3.2-3B-Instruct --device cuda:0 --states_dir ./states --user_id 0 --timeout_sec 15"

    print("🧪 STARTING SPD FRAMEWORK COMPARISON TEST")
    print("This will run SPD with and without Freivalds integration")

    # Check if states directory exists
    if not os.path.exists("./states"):
        print("❌ Error: ./states directory not found!")
        print("Please run prompt obfuscation first:")
        print("python3 po.py --prompt input.txt --gamma 5 --epsilon 0.1 --temperature 1.0 --prob_dist abs --model meta-llama/Llama-3.2-3B-Instruct --device cuda:0 --states_dir ./states --verbose")
        return

    # Run experiment WITHOUT Freivalds
    print("\n🔬 PHASE 1: Testing SPD WITHOUT Freivalds integration")
    results_without = run_experiment(
        "SPD WITHOUT Freivalds",
        "smd_without_freivalds.py",
        master_cmd.replace("smd.py", "smd_without_freivalds.py"),
        worker_cmd_without,
        timeout=60
    )

    print("\n⏳ Waiting 5 seconds before next test...")
    time.sleep(5)

    # Run experiment WITH Freivalds
    print("\n🔬 PHASE 2: Testing SPD WITH Freivalds integration")
    results_with = run_experiment(
        "SPD WITH Freivalds",
        "smd.py",
        master_cmd,
        worker_cmd_with,
        timeout=60
    )

    # Analyze and compare results
    analyze_results(results_without, results_with)

    # Save detailed results
    with open("comparison_results.txt", "w", encoding="utf-8") as f:
        f.write("SPD FRAMEWORK COMPARISON RESULTS\n")
        f.write("="*50 + "\n\n")

        f.write("WITHOUT FREIVALDS:\n")
        f.write(f"Success: {results_without['success']}\n")
        f.write(f"Master Time: {results_without['master_time']:.2f}s\n")
        f.write(f"Master Output Length: {len(results_without['master_output'])}\n\n")

        f.write("WITH FREIVALDS:\n")
        f.write(f"Success: {results_with['success']}\n")
        f.write(f"Master Time: {results_with['master_time']:.2f}s\n")
        f.write(f"Master Output Length: {len(results_with['master_output'])}\n\n")

        f.write("DETAILED OUTPUTS:\n\n")
        f.write("WITHOUT FREIVALDS MASTER:\n")
        f.write(results_without['master_output'])
        f.write("\n\nWITHOUT FREIVALDS WORKER:\n")
        f.write(results_without['worker_output'])

        f.write("\n\nWITH FREIVALDS MASTER:\n")
        f.write(results_with['master_output'])
        f.write("\n\nWITH FREIVALDS WORKER:\n")
        f.write(results_with['worker_output'])

    print("\n📄 Detailed results saved to: comparison_results.txt")

if __name__ == "__main__":
    main()
