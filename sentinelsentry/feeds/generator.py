"""
RSS 2.0 XML generator for simulated dispatch & crisis streams.
"""

from __future__ import annotations

import email.utils
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from xml.sax.saxutils import escape

from sentinelsentry.schema import RawFeedItem
from sentinelsentry.feeds.mock_streams import CRISIS_DISPATCH_SCENARIOS, get_feed_stream_items


class RSSFeedGenerator:
    """
    Generates standard RSS 2.0 XML streams for emergency dispatch feeds.
    """

    def __init__(
        self,
        feed_id: str,
        feed_name: str,
        description: str,
        feed_link: str = "http://sentinelsentry.local/feeds",
    ):
        self.feed_id = feed_id
        self.feed_name = feed_name
        self.description = description
        self.feed_link = feed_link
        self._items: List[Dict[str, Any]] = []

    def populate_from_scenarios(self) -> None:
        """Seed this feed generator with matching items from standard scenarios."""
        matching = get_feed_stream_items(self.feed_id)
        for item in matching:
            self.add_item(
                title=item.get("title", f"Dispatch Alert: {self.feed_id}"),
                raw_text=item["raw_text"],
                published_at=datetime.now(timezone.utc),
                scenario_id=item.get("scenario_id"),
            )

    def add_item(
        self,
        raw_text: str,
        title: Optional[str] = None,
        guid: Optional[str] = None,
        published_at: Optional[datetime] = None,
        link: Optional[str] = None,
        scenario_id: Optional[str] = None,
    ) -> str:
        """Add an incident item to the feed and return its GUID."""
        pub_date = published_at or datetime.now(timezone.utc)
        if not guid:
            # Deterministic unique hash based on feed, content, and timestamp
            content_sig = f"{self.feed_id}:{raw_text}:{pub_date.isoformat()}"
            guid = hashlib.sha256(content_sig.encode("utf-8")).hexdigest()[:16]

        if not title:
            # Create informative headline
            title = f"[{self.feed_id.upper()}] " + (raw_text[:60] + "..." if len(raw_text) > 60 else raw_text)

        if not link:
            link = f"{self.feed_link}/{self.feed_id}/item/{guid}"

        item_dict = {
            "guid": guid,
            "title": title,
            "raw_text": raw_text,
            "pub_date": pub_date,
            "link": link,
            "scenario_id": scenario_id,
        }
        self._items.append(item_dict)
        return guid

    def get_items(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        items = sorted(self._items, key=lambda x: x["pub_date"], reverse=True)
        if limit:
            items = items[:limit]
        return items

    def to_raw_feed_items(self, limit: Optional[int] = None) -> List[RawFeedItem]:
        """Convert internal items to RawFeedItem models."""
        items = self.get_items(limit)
        return [
            RawFeedItem(
                id=item["guid"],
                feed_id=self.feed_id,
                feed_name=self.feed_name,
                title=item["title"],
                raw_text=item["raw_text"],
                published_at=item["pub_date"],
                link=item["link"],
                guid=item["guid"],
                metadata={"scenario_id": item.get("scenario_id")},
            )
            for item in items
        ]

    def to_rss_xml(self, limit: int = 50) -> str:
        """
        Generate strict, valid RSS 2.0 XML string.
        """
        now_rfc822 = email.utils.format_datetime(datetime.now(timezone.utc))
        items = self.get_items(limit)

        xml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">',
            "  <channel>",
            f"    <title>{escape(self.feed_name)}</title>",
            f"    <link>{escape(self.feed_link)}</link>",
            f"    <description>{escape(self.description)}</description>",
            "    <language>en-us</language>",
            f"    <pubDate>{now_rfc822}</pubDate>",
            f'    <atom:link href="{escape(self.feed_link)}/{escape(self.feed_id)}.xml" rel="self" type="application/rss+xml" />',
        ]

        for it in items:
            it_rfc822 = email.utils.format_datetime(it["pub_date"])
            xml_lines.append("    <item>")
            xml_lines.append(f"      <title>{escape(it['title'])}</title>")
            xml_lines.append(f"      <link>{escape(it['link'])}</link>")
            xml_lines.append(f"      <description>{escape(it['raw_text'])}</description>")
            xml_lines.append(f"      <pubDate>{it_rfc822}</pubDate>")
            xml_lines.append(f'      <guid isPermaLink="false">{escape(it["guid"])}</guid>')
            if it.get("scenario_id"):
                xml_lines.append(f"      <category>{escape(it['scenario_id'])}</category>")
            xml_lines.append("    </item>")

        xml_lines.append("  </channel>")
        xml_lines.append("</rss>")
        return "\n".join(xml_lines)
