from statsbombpy import sb
import pandas as pd


def list_competitions() -> pd.DataFrame:
    """Return all available StatsBomb Open Data competitions."""
    return sb.competitions()


def list_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    """Return all matches for a competition/season."""
    return sb.matches(competition_id=competition_id, season_id=season_id)


def get_events(match_id: int) -> pd.DataFrame:
    """Return all events for a match as a flat DataFrame."""
    return sb.events(match_id=match_id)


def get_360(match_id: int) -> pd.DataFrame:
    """Return 360 freeze-frame positional data for a match."""
    return sb.frames(match_id=match_id)
