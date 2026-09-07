"""
search.py
---------
Step 2 of the pipeline: take the face image and run a REAL reverse-image
search against the web to find a matching social media post.

This module provides multiple live search engines and a robust fallback chain:
1. SerpAPI (Google Lens / Google Reverse Image) - via direct file upload or URL
2. Yandex Reverse Image Search - keyless, zero API key required, supports direct local file upload
3. Bing Visual Search (Azure Cognitive Services) - full multipart upload with PagesIncluding + VisualSearch
4. Free anonymous image hosting fallback (tmpfiles.org / 0x0.st) if a public URL is needed

Interface:
  search(image_path, provider="auto", max_results=5) -> List[SearchResult]
"""

import os
import json
import logging
import urllib.parse
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass
class SearchResult:
    source_url: str          # the webpage/post where the image appears
    platform: Optional[str]  # best-effort guess: instagram/twitter/facebook/linkedin/reddit/web
    title: Optional[str]     # title / snippet of the post or page
    matched_image_url: Optional[str] # URL to the candidate thumbnail/image
    snippet: Optional[str] = None
    provider: Optional[str] = "web"


class SearchProviderError(Exception):
    """Raised when an external reverse image search provider fails."""
    pass


def _guess_platform(url: str) -> str:
    """Classifies URL into common social media platforms or web domain."""
    url_lower = (url or "").lower()
    for name in ["instagram", "twitter", "x.com", "facebook", "linkedin", "tiktok", "reddit", "pinterest", "youtube", "github"]:
        if name in url_lower:
            return name.replace(".com", "")
    try:
        parsed = urllib.parse.urlparse(url)
        netloc = parsed.netloc.replace("www.", "")
        return netloc if netloc else "web"
    except Exception:
        return "web"


def upload_temp_image(image_path: str) -> str:
    """
    Uploads a local image to a free anonymous host (tmpfiles.org or 0x0.st)
    to generate a publicly reachable URL if an external API requires it.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Try tmpfiles.org
    try:
        with open(image_path, "rb") as f:
            resp = requests.post("https://tmpfiles.org/api/v1/upload", files={"file": f}, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            url = data.get("data", {}).get("url")
            if url:
                # tmpfiles direct download URL requires /dl/ inserted
                direct_url = url.replace("tmpfiles.org/", "tmpfiles.org/dl/")
                return direct_url
    except Exception as e:
        logger.warning(f"tmpfiles.org upload failed: {e}")

    # Fallback to 0x0.st
    try:
        with open(image_path, "rb") as f:
            resp = requests.post("https://0x0.st", files={"file": f}, timeout=15)
        if resp.status_code == 200 and resp.text.startswith("http"):
            return resp.text.strip()
    except Exception as e:
        logger.warning(f"0x0.st upload failed: {e}")

    raise RuntimeError("Failed to host temporary image on public providers.")


class SerpApiReverseImageSearch:
    """
    SerpAPI provider supporting Google Lens and Google Reverse Image.
    Free tier: 100 searches/month.
    """
    ENDPOINT = "https://serpapi.com/search.json"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("SERPAPI_KEY")

    def is_available(self) -> bool:
        return bool(self.api_key or os.environ.get("SERPAPI_KEY"))

    def search_by_file(self, image_path: str, max_results: int = 5) -> List[SearchResult]:
        key = self.api_key or os.environ.get("SERPAPI_KEY")
        if not key:
            raise SearchProviderError("No SerpApi key found. Set SERPAPI_KEY env var or in .env.")
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Upload local file to temporary host to obtain direct URL for SerpAPI
        try:
            public_url = upload_temp_image(image_path)
        except Exception as ex:
            raise SearchProviderError(f"Could not host image for SerpAPI: {ex}")

        return self.search_by_image_url(public_url, max_results=max_results)

    def search_by_image_url(self, image_url: str, max_results: int = 5) -> List[SearchResult]:
        key = self.api_key or os.environ.get("SERPAPI_KEY")
        if not key:
            raise SearchProviderError("No SerpApi key found. Set SERPAPI_KEY env var or in .env.")

        # 1. Try Google Lens engine first
        params = {
            "engine": "google_lens",
            "url": image_url,
            "api_key": key,
        }
        try:
            resp = requests.get(self.ENDPOINT, params=params, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                if "error" in data:
                    raise SearchProviderError(f"SerpAPI error: {data['error']}")
                results = []
                for item in data.get("visual_matches", [])[:max_results]:
                    link = item.get("link") or item.get("source")
                    cand_url = item.get("thumbnail") or item.get("original")
                    if not link:
                        continue
                    results.append(SearchResult(
                        source_url=link,
                        platform=_guess_platform(link),
                        title=item.get("title", "Google Lens Visual Match"),
                        matched_image_url=cand_url,
                        snippet=item.get("snippet"),
                        provider="serpapi_lens",
                    ))
                if results:
                    return results

            # 2. Fallback to Google Reverse Image engine
            params = {
                "engine": "google_reverse_image",
                "image_url": image_url,
                "api_key": key,
            }
            resp = requests.get(self.ENDPOINT, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                raise SearchProviderError(f"SerpAPI error: {data['error']}")

            results = []
            for item in data.get("image_results", [])[:max_results]:
                link = item.get("link") or item.get("source")
                if not link:
                    continue
                results.append(SearchResult(
                    source_url=link,
                    platform=_guess_platform(link),
                    title=item.get("title"),
                    matched_image_url=item.get("thumbnail") or item.get("original"),
                    snippet=item.get("snippet"),
                    provider="serpapi_reverse_image",
                ))
            return results
        except requests.RequestException as e:
            raise SearchProviderError(f"SerpAPI search request failed: {e}")


class YandexReverseImageSearch:
    """
    Keyless scripted Yandex reverse image search.
    Uploads local image directly to Yandex Images cbir endpoint and scrapes candidate matches.
    Requires no API keys and works out of the box.
    """
    UPLOAD_URL = "https://yandex.com/images/search"

    def search_by_file(self, image_path: str, max_results: int = 5) -> List[SearchResult]:
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        session = requests.Session()
        session.headers.update(DEFAULT_HEADERS)

        try:
            with open(image_path, "rb") as f:
                files = {"upfile": ("image.jpg", f, "image/jpeg")}
                params = {
                    "rpt": "imageview",
                    "format": "json",
                    "request": json.dumps({"blocks": [{"block": "b-page_type_search-by-image__link"}]}),
                }
                res = session.post(self.UPLOAD_URL, params=params, files=files, timeout=25)

            if res.status_code != 200:
                raise SearchProviderError(f"Yandex upload returned HTTP {res.status_code}")

            try:
                payload = res.json()
                cbir_id = payload["blocks"][0]["params"].get("cbirId")
                if not cbir_id:
                    query_url = payload["blocks"][0]["params"].get("url")
                    if not query_url:
                        raise KeyError("cbirId and url not found in response")
                    results_url = f"https://yandex.com/images/search?{query_url}"
                else:
                    results_url = f"https://yandex.com/images/search?cbir_id={urllib.parse.quote(cbir_id)}&rpt=imageview"
            except (KeyError, IndexError, json.JSONDecodeError) as exc:
                if "captcha" in res.text.lower() or "smartcaptcha" in res.text.lower():
                    raise SearchProviderError("Yandex bot verification / SmartCaptcha encountered.")
                raise SearchProviderError(f"Unexpected response structure from Yandex: {exc}")

            # Fetch results page
            results_resp = session.get(results_url, timeout=25)
            if results_resp.status_code != 200:
                raise SearchProviderError(f"Yandex results page returned HTTP {results_resp.status_code}")

            soup = BeautifulSoup(results_resp.text, "html.parser")
            results = []

            # 1. Parse .CbirSites-Item elements (pages featuring the image)
            for item in soup.select(".CbirSites-Item"):
                if len(results) >= max_results:
                    break
                link_tag = item.find("a", href=True)
                img_tag = item.find("img", src=True)
                title_tag = item.select_one(".CbirSites-ItemTitle, .CbirSites-ItemDescription")
                title = title_tag.get_text(strip=True) if title_tag else "Web Post"

                source_url = link_tag["href"] if link_tag else results_url
                img_src = img_tag["src"] if img_tag else None
                if img_src and img_src.startswith("//"):
                    img_src = "https:" + img_src

                results.append(SearchResult(
                    source_url=source_url,
                    platform=_guess_platform(source_url),
                    title=title,
                    matched_image_url=img_src,
                    provider="yandex",
                ))

            # 2. Parse similar thumbnails if CbirSites is empty
            if len(results) < max_results:
                for thumb in soup.select(".Thumb-Image, img[src*='images-thumbs']"):
                    if len(results) >= max_results:
                        break
                    src = thumb.get("src")
                    if src:
                        if src.startswith("//"):
                            src = "https:" + src
                        if not any(r.matched_image_url == src for r in results):
                            parent_a = thumb.find_parent("a", href=True)
                            page_link = parent_a["href"] if parent_a else results_url
                            if page_link.startswith("/"):
                                page_link = "https://yandex.com" + page_link
                            results.append(SearchResult(
                                source_url=page_link,
                                platform=_guess_platform(page_link),
                                title="Similar Image Match",
                                matched_image_url=src,
                                provider="yandex",
                            ))

            return results

        except requests.RequestException as e:
            raise SearchProviderError(f"Yandex connection failed: {e}")


class BingVisualSearch:
    """
    Microsoft Azure Bing Visual Search API.
    Uploads local image directly to Azure Cognitive Services.
    """
    ENDPOINT = os.environ.get("BING_ENDPOINT", "https://api.bing.microsoft.com/v7.0/images/visualsearch")

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("BING_VISUAL_SEARCH_KEY") or os.environ.get("BING_API_KEY")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def search_by_file(self, image_path: str, max_results: int = 5) -> List[SearchResult]:
        if not self.api_key:
            raise SearchProviderError("No Bing key found. Set BING_VISUAL_SEARCH_KEY env var.")
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        try:
            with open(image_path, "rb") as f:
                files = {"image": ("query.jpg", f, "image/jpeg")}
                resp = requests.post(self.ENDPOINT, headers=headers, files=files, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            results = []
            for tag in data.get("tags", []):
                for action in tag.get("actions", []):
                    action_type = action.get("actionType", "")
                    if action_type in ("VisualSearch", "PagesIncluding"):
                        for v in action.get("data", {}).get("value", [])[:max_results]:
                            host_url = v.get("hostPageUrl") or v.get("webSearchUrl") or ""
                            content_url = v.get("contentUrl") or v.get("thumbnailUrl")
                            if host_url or content_url:
                                results.append(SearchResult(
                                    source_url=host_url or content_url,
                                    platform=_guess_platform(host_url),
                                    title=v.get("name", "Bing Visual Match"),
                                    matched_image_url=content_url,
                                    provider="bing",
                                ))
            return results[:max_results]
        except requests.RequestException as e:
            raise SearchProviderError(f"Bing Visual Search failed: {e}")


def search(image_path: str, provider: str = "auto", max_results: int = 5) -> List[SearchResult]:
    """
    High-level search entry point with automatic provider selection and fallback.

    Providers:
      - 'auto' / 'chain': Tries SerpAPI (if key present) -> Yandex (keyless) -> Bing (if key present)
      - 'serpapi': Uses SerpAPI Google Lens
      - 'yandex': Uses Keyless Yandex Reverse Image search
      - 'bing': Uses Azure Bing Visual Search
    """
    prov = (provider or "auto").strip().lower()

    if prov == "serpapi":
        return SerpApiReverseImageSearch().search_by_file(image_path, max_results=max_results)
    elif prov == "yandex":
        return YandexReverseImageSearch().search_by_file(image_path, max_results=max_results)
    elif prov == "bing":
        return BingVisualSearch().search_by_file(image_path, max_results=max_results)

    # Automatic / Chain Fallback
    errors = []

    # 1. SerpAPI (if configured)
    serp = SerpApiReverseImageSearch()
    if serp.is_available():
        try:
            results = serp.search_by_file(image_path, max_results=max_results)
            if results:
                return results
        except Exception as e:
            errors.append(f"SerpAPI: {e}")

    # 2. Yandex (keyless scripted request)
    try:
        yandex = YandexReverseImageSearch()
        results = yandex.search_by_file(image_path, max_results=max_results)
        if results:
            return results
    except Exception as e:
        errors.append(f"Yandex: {e}")

    # 3. Bing (if configured)
    bing = BingVisualSearch()
    if bing.is_available():
        try:
            results = bing.search_by_file(image_path, max_results=max_results)
            if results:
                return results
        except Exception as e:
            errors.append(f"Bing: {e}")

    error_summary = " | ".join(errors) if errors else "No results returned."
    raise SearchProviderError(f"All search providers in chain failed or found no results: {error_summary}")