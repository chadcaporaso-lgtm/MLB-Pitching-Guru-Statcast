import os
import re
import json
import requests
import feedparser
from datetime import datetime

BASE_DIR = '/content/drive/MyDrive/MLB-Playoffs-Data'
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_PATH = os.path.join(DATA_DIR, 'manager_context_overrides.json')

RSS_FEEDS = [
    'https://www.mlbtraderumors.com/feed',
    'https://www.rotowire.com/rss/news.php?sport=mlb'
]

def get_slate_probable_pitchers(date_str: str = None) -> list:
    if not date_str:
        date_str = datetime.now().strftime('%Y-%m-%d')
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}&hydrate=probablePitcher"
    starters = []
    try:
        res = requests.get(url, timeout=10)
        data = res.json()
        for date_obj in data.get('dates', []):
            for game in date_obj.get('games', []):
                away_p = game.get('teams', {}).get('away', {}).get('probablePitcher', {}).get('fullName')
                home_p = game.get('teams', {}).get('home', {}).get('probablePitcher', {}).get('fullName')
                game_pk = game.get('gamePk')
                if away_p:
                    starters.append({'name': away_p, 'team': game['teams']['away']['team']['name'], 'gamePk': game_pk})
                if home_p:
                    starters.append({'name': home_p, 'team': game['teams']['home']['team']['name'], 'gamePk': game_pk})
    except Exception as e:
        print(f"[-] Error fetching probable pitchers from MLB Stats API: {e}")
    return starters

def fetch_mlb_editorial_content(game_pk: int) -> str:
    url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/content"
    try:
        res = requests.get(url, timeout=8)
        data = res.json()
        items = data.get('editorial', {}).get('preview', {}).get('items', [])
        return " ".join([i.get('body', '') for i in items if 'body' in i])
    except Exception:
        return ""

def fetch_beat_rss_corpus() -> list:
    articles = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    for feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url, request_headers=headers)
            for entry in feed.entries[:30]:
                title = entry.get('title', '')
                summary = entry.get('summary', '')
                articles.append(f"{title}: {summary}")
        except Exception as e:
            print(f"[-] RSS feed parse warning for {feed_url}: {e}")
    return articles

def extract_pitcher_constraints(pitcher_name: str, corpus_texts: list) -> dict:
    escaped_name = re.escape(pitcher_name)
    cap_pattern = rf"{escaped_name}.*?(\d{{2,3}})(?:\s*-\s*\d{{2,3}})?\s*(?:pitches|pitch count|pitch limit)"
    tuneup_pattern = rf"{escaped_name}.*?(?:tune-?up|short leash|brief outing|few innings|prep for playoffs|rested for wc)"
    workhorse_pattern = rf"{escaped_name}.*?(?:must-win|elimination|win or go home|season on the line|untethered|full workload)"

    combined_text = " ".join([t for t in corpus_texts if pitcher_name.lower() in t.lower()])
    if not combined_text:
        return None

    if re.search(workhorse_pattern, combined_text, re.IGNORECASE):
        return {
            'pitch_cap': 105,
            'tto_limit': 4,
            'leash_type': 'MUST_WIN_WORKHORSE',
            'reason': 'Elimination / must-win context identified in beat coverage'
        }

    cap_match = re.search(cap_pattern, combined_text, re.IGNORECASE)
    if cap_match:
        cap_val = int(cap_match.group(1))
        return {
            'pitch_cap': cap_val,
            'tto_limit': 2 if cap_val <= 75 else 3,
            'leash_type': 'EXPLICIT_BEAT_CAP',
            'reason': f"Explicit limit found in quotes: {cap_val} pitches"
        }

    if re.search(tuneup_pattern, combined_text, re.IGNORECASE):
        return {
            'pitch_cap': 65,
            'tto_limit': 2,
            'leash_type': 'TUNE_UP_SHORT_LEASH',
            'reason': 'Playoff preparation / tune-up language identified'
        }

    return None

def run_beat_harvester(date_str: str = None):
    print(f"[*] Starting Beat News Harvester for {date_str or datetime.now().strftime('%Y-%m-%d')}...")
    starters = get_slate_probable_pitchers(date_str)
    print(f"[*] Identified {len(starters)} active starting arms from MLB API.")

    rss_articles = fetch_beat_rss_corpus()
    overrides = {}

    for s in starters:
        name = s['name']
        game_pk = s.get('gamePk')
        editorial = fetch_mlb_editorial_content(game_pk) if game_pk else ""
        corpus = rss_articles + ([editorial] if editorial else [])
        
        result = extract_pitcher_constraints(name, corpus)
        if result:
            overrides[name] = result
            overrides[name]['updated_at'] = datetime.utcnow().isoformat()
            print(f"    [+] OVERRIDE FOUND: {name} -> {result['pitch_cap']} pitches ({result['leash_type']})")

    # Hard guardrail for Zack Wheeler elimination game
    if 'Zack Wheeler' in [s['name'] for s in starters] and 'Zack Wheeler' not in overrides:
        overrides['Zack Wheeler'] = {
            'pitch_cap': 105,
            'tto_limit': 4,
            'leash_type': 'MUST_WIN_WORKHORSE',
            'reason': 'Phillies Game 162 elimination scenario: uncapped workhorse leash.',
            'updated_at': datetime.utcnow().isoformat()
        }
        print("    [+] GUARDRAIL INJECTED: Zack Wheeler -> 105 pitches (MUST_WIN_WORKHORSE)")

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(overrides, f, indent=2)
    print(f"[✓] Overrides file saved to {OUTPUT_PATH} ({len(overrides)} total active constraints).")

if __name__ == '__main__':
    run_beat_harvester()
