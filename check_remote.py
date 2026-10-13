import asyncio, sys
from mcp import Client

URL = sys.argv[1]

async def main():
    async with Client(URL) as c:
        print("connected to:", c.server_info.name)
        print("protocol:", c.protocol_version)
        tools = await c.list_tools()
        print("tools:", [t.name for t in tools.tools])
        result = await c.call_tool("get_stuck_workflows", {})
        print(result.content[0].text[:200])

asyncio.run(main())