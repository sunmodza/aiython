"""A standalone AI statement can update objects in the live Python frame."""
tasks = [
    {"id": 1, "note": "The release notes are complete", "done": False},
    {"id": 2, "note": "The security review is still pending", "done": False},
]
original_tasks = tasks

review each note in tasks and set done to True only for completed tasks

assert tasks is original_tasks
print(tasks)
