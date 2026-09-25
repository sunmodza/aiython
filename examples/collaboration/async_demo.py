"""Run with: uv run aiython examples/collaboration/async_demo.py"""
import asyncio

from aiython import group, join


async def worker(ticket):
    async with join(ticket) as me:
        message = (await me.wait_async(timeout=5))[0]
        me.send("main", {"answer": message["payload"] * 2})


async def main():
    with group() as team:
        ticket = team.invite("worker")
        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(worker(ticket))
            team.send("worker", 21)
        print(team.read()[0]["payload"])


if __name__ == "__main__":
    asyncio.run(main())
