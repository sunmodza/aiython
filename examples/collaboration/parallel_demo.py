"""Run with: uv run aiython examples/collaboration/parallel_demo.py"""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import get_context
from pathlib import Path

from aiython import group, worker_entry


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    with group(project_root) as team:
        first = team.invite("thread-worker")
        second = team.invite("process-worker")
        with ThreadPoolExecutor(max_workers=1) as threads:
            thread = threads.submit(worker_entry, first,
                                    "examples.collaboration.worker", "process_item", 3)
            with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as processes:
                process = processes.submit(worker_entry, second,
                                           "examples.collaboration.worker", "process_item", 4)
                print("results:", thread.result(), process.result())
                print("messages:", [message["payload"] for message in team.read()])
