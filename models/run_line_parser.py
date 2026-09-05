"""
Run Line (+1.5 / -1.5) Dynamic Attribute Parser
------------------------------------------------
Inspects explicit bookmaker spread attributes directly from SportsGameOdds (SGO)
API payloads to eliminate pick'em inversion bugs and prevent pairing underdog 
probabilities with favorite plus-money payouts.
"""

from typing import Dict, List, Any, Optional

def to_decimal(amer: Any) -> float:
    try:
        val = float(amer)
        if val > 0:
            return round(1.0 + (val / 100.0), 4)
        elif val < 0:
            return round(1.0 + (100.0 / abs(val)), 4)
        return 1.0
    except (ValueError, TypeError):
        return 1.0

def to_american(prob: float) -> str:
    if prob >= 0.999:
        return "-10000"
    if prob <= 0.001:
        return "+10000"
    if prob >= 0.5:
        return f"-{int(round(100.0 * (prob / (1.0 - prob))))}"
    return f"+{int(round(100.0 * ((1.0 - prob) / prob)))}"

class RunLineParser:
    def __init__(self, target_books: Optional[List[str]] = None):
        self.target_books = target_books or ["novig", "fanduel", "caesars", "draftkings", "betmgm"]

    def parse_game_run_lines(
        self,
        event: Dict[str, Any],
        model_home_m15_prob: float,
        model_away_p15_prob: float
    ) -> List[Dict[str, Any]]:
        """
        Parses home and away spread trees by extracting the explicit 'spread' attribute
        from each bookmaker rather than relying solely on oddID conventions.
        """
        home_obj = event.get("teams", {}).get("home", {})
        away_obj = event.get("teams", {}).get("away", {})

        home_name = (
            home_obj.get("names", {}).get("long")
            or home_obj.get("names", {}).get("medium")
            or home_obj.get("teamID", "Home")
        )
        away_name = (
            away_obj.get("names", {}).get("long")
            or away_obj.get("names", {}).get("medium")
            or away_obj.get("teamID", "Away")
        )
        matchup_label = f"{away_name} @ {home_name}"
        odds_tree = event.get("odds", {})

        # Probability mapping based on true game condition:
        # If Home -1.5 is prob P, Home +1.5 is 1 - P.
        # If Away +1.5 is prob Q, Away -1.5 is 1 - Q.
        probability_map = {
            (home_name, "-1.5"): float(model_home_m15_prob),
            (home_name, "+1.5"): round(1.0 - float(model_home_m15_prob), 4),
            (away_name, "-1.5"): round(1.0 - float(model_away_p15_prob), 4),
            (away_name, "+1.5"): float(model_away_p15_prob),
        }

        # Market buckets keyed by (TeamName, SpreadString)
        buckets: Dict[tuple, Dict[str, int]] = {
            (home_name, "-1.5"): {},
            (home_name, "+1.5"): {},
            (away_name, "-1.5"): {},
            (away_name, "+1.5"): {},
        }

        market_targets = [
            ("points-home-game-sp-home", home_name),
            ("points-away-game-sp-away", away_name)
        ]

        for odd_id, entity_name in market_targets:
            by_book = odds_tree.get(odd_id, {}).get("byBookmaker", {})
            for b_name, b_data in by_book.items():
                if not b_data.get("available", True):
                    continue

                raw_odds = b_data.get("odds")
                raw_spread = b_data.get("spread")

                if raw_odds is None or raw_spread is None:
                    continue

                try:
                    sp_float = float(raw_spread)
                    sp_str = f"+{sp_float:.1f}" if sp_float > 0 else f"{sp_float:.1f}"
                    odds_int = int(raw_odds)
                    
                    bucket_key = (entity_name, sp_str)
                    if bucket_key in buckets:
                        buckets[bucket_key][b_name.lower()] = odds_int
                except (ValueError, TypeError):
                    continue

        results = []
        for (team, sp_str), book_quotes in buckets.items():
            if not book_quotes:
                continue

            filtered_quotes = {
                b: odds for b, odds in book_quotes.items()
                if b in self.target_books
            }
            if not filtered_quotes:
                filtered_quotes = book_quotes

            best_book = max(filtered_quotes.keys(), key=lambda b: to_decimal(filtered_quotes[b]))
            best_odds = filtered_quotes[best_book]
            win_prob = probability_map.get((team, sp_str), 0.500)
            ev = round((win_prob * to_decimal(best_odds)) - 1.0, 4)

            results.append({
                "matchup": matchup_label,
                "team": team,
                "spread": sp_str,
                "selection": f"{team} {sp_str}",
                "model_prob": win_prob,
                "fair_line": to_american(win_prob),
                "best_book": best_book,
                "best_odds": best_odds,
                "ev": ev,
                "quotes": book_quotes
            })

        return sorted(results, key=lambda x: x["ev"], reverse=True)
