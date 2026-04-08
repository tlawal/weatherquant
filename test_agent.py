import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.market_context.service import refresh_market_context_snapshot

async def main():
    print("Triggering agentic Market Context refresh for Atlanta...")
    result = await refresh_market_context_snapshot("atlanta", "2026-04-06")
    print("\nRefresh Complete. Result:")
    print(result.get("generation_status"))
    if result.get("last_error"):
        print("Error:", result.get("last_error"))
    print("\nSections snippet:")
    sections = result.get("sections")
    if sections:
        for k, v in sections.items():
            print(f"[{k}] {v[:100]}...")
            
    print("\nSelection snippet:")
    selection = result.get("selection")
    if selection:
        for k, v in selection.items():
            print(f"{k}: {v}")

if __name__ == "__main__":
    asyncio.run(main())
