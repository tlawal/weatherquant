import asyncio
from sqlalchemy import select
from backend.storage.db import get_session
from backend.storage.models import City, Event, Bucket

async def main():
    async with get_session() as sess:
        result = await sess.execute(select(City).where(City.city_slug == "seattle"))
        city = result.scalar_one_or_none()
        if not city:
            print("No seattle city")
            return
        result = await sess.execute(select(Event).where(Event.city_id == city.id).order_by(Event.date_et.desc()).limit(1))
        event = result.scalar_one_or_none()
        if not event:
            print("No events")
            return
        result = await sess.execute(select(Bucket).where(Bucket.event_id == event.id).order_by(Bucket.bucket_idx))
        for b in result.scalars():
            print(f"[{b.bucket_idx}] {b.label} | low={b.low_f} high={b.high_f}")

asyncio.run(main())
